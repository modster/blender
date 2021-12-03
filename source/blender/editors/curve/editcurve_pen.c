/*
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

/** \file
 * \ingroup edcurve
 */

#include "DNA_curve_types.h"

#include "MEM_guardedalloc.h"

#include "BLI_listbase.h"
#include "BLI_math.h"

#include "BKE_context.h"
#include "BKE_curve.h"

#include "DEG_depsgraph.h"

#include "WM_api.h"

#include "ED_curve.h"
#include "ED_screen.h"
#include "ED_view3d.h"

#include "BKE_object.h"

#include "curve_intern.h"

#include "RNA_access.h"
#include "RNA_define.h"

#include "float.h"

/* Data structure to keep track of details about the cut location */
typedef struct CutData {
  /* Index of the last #BezTriple or BPoint before the cut. */
  int bezt_index, bp_index;
  /* Nurb to which the cut belongs to. */
  Nurb *nurb;
  /* Minimum distance to curve from mouse location. */
  float min_dist;
  /* Fraction of segments after which the new point divides the curve segment. */
  float parameter;
  /* Whether the currently identified closest point has any vertices before/after it. */
  bool has_prev, has_next;
  /* Locations of adjacent vertices and cut location. */
  float prev_loc[3], cut_loc[3], next_loc[3];
  /* Mouse location in floats. */
  float mval[2];
} CutData;

/* Data required for segment altering functionality. */
typedef struct MoveSegmentData {
  /* Nurb being altered. */
  Nurb *nu;
  /* Index of the #BezTriple before the segment. */
  int bezt_index;
} MoveSegmentData;

static void mouse_location_to_worldspace(const int mouse_loc[2],
                                         const float depth[3],
                                         const ViewContext *vc,
                                         float r_location[3])
{
  mul_v3_m4v3(r_location, vc->obedit->obmat, depth);
  ED_view3d_win_to_3d_int(vc->v3d, vc->region, r_location, mouse_loc, r_location);

  float imat[4][4];
  invert_m4_m4(imat, vc->obedit->obmat);
  mul_m4_v3(imat, r_location);
}

/* Move the handle of the newly added #BezTriple to mouse. */
static void move_bezt_handles_to_mouse(BezTriple *bezt,
                                       const bool is_end_point,
                                       const wmEvent *event,
                                       const ViewContext *vc)
{
  if (bezt->h1 == HD_VECT && bezt->h2 == HD_VECT) {
    bezt->h1 = HD_ALIGN;
    bezt->h2 = HD_ALIGN;
  }

  float location[3];
  mouse_location_to_worldspace(event->mval, bezt->vec[1], vc, location);

  /* If the new point is the last point of the curve, move the second handle to the mouse. */
  if (is_end_point) {
    copy_v3_v3(bezt->vec[2], location);

    if (bezt->h2 != HD_FREE) {
      mul_v3_fl(location, -1.0f);
      madd_v3_v3v3fl(bezt->vec[0], location, bezt->vec[1], 2.0f);
    }
  }
  else {
    copy_v3_v3(bezt->vec[0], location);

    if (bezt->h1 != HD_FREE) {
      mul_v3_fl(location, -1.0f);
      madd_v3_v3v3fl(bezt->vec[2], location, bezt->vec[1], 2.0f);
    }
  }
}

/* Move entire control point to given worldspace location. */
static void move_bezt_to_location(BezTriple *bezt, const float location[3])
{
  float change[3];
  sub_v3_v3v3(change, location, bezt->vec[1]);
  add_v3_v3(bezt->vec[0], change);
  copy_v3_v3(bezt->vec[1], location);
  add_v3_v3(bezt->vec[2], change);
}

/* Alter handle types to allow free movement (Set handles to #FREE or #ALIGN). */
static void free_up_handles_for_movement(BezTriple *bezt, const bool f1, const bool f3)
{
  if (f1) {
    if (bezt->h1 == HD_VECT) {
      bezt->h1 = HD_FREE;
    }
    if (bezt->h1 == HD_AUTO) {
      bezt->h1 = HD_ALIGN;
      bezt->h2 = HD_ALIGN;
    }
  }
  if (f3) {
    if (bezt->h2 == HD_VECT) {
      bezt->h2 = HD_FREE;
    }
    if (bezt->h2 == HD_AUTO) {
      bezt->h1 = HD_ALIGN;
      bezt->h2 = HD_ALIGN;
    }
  }
}

/* Move handles or entire #BezTriple to mouse based on selection. */
static void move_selected_bezt_to_mouse(BezTriple *bezt,
                                        const ViewContext *vc,
                                        const wmEvent *event)
{
  float location[3];
  if (BEZT_ISSEL_IDX(bezt, 1)) {
    mouse_location_to_worldspace(event->mval, bezt->vec[1], vc, location);
    move_bezt_to_location(bezt, location);
  }
  else {
    free_up_handles_for_movement(bezt, bezt->f1, bezt->f3);
    if (BEZT_ISSEL_IDX(bezt, 0)) {
      mouse_location_to_worldspace(event->mval, bezt->vec[0], vc, location);
      copy_v3_v3(bezt->vec[0], location);
    }
    else {
      mouse_location_to_worldspace(event->mval, bezt->vec[2], vc, location);
      copy_v3_v3(bezt->vec[2], location);
    }
  }
}

static void move_bp_to_mouse(BPoint *bp, const wmEvent *event, const ViewContext *vc)
{
  float location[3];
  mouse_location_to_worldspace(event->mval, bp->vec, vc, location);

  copy_v3_v3(bp->vec, location);
}

static int get_nurb_index(const ListBase *nurbs, const Nurb *nurb)
{
  int index = 0;
  LISTBASE_FOREACH (Nurb *, nu, nurbs) {
    if (nu == nurb) {
      return index;
    }
    index++;
  }
  /* The Nurb should've been found by now. */
  BLI_assert(false);
  return -1;
}

static void delete_nurb(Curve *cu, Nurb *nu)
{
  EditNurb *editnurb = cu->editnurb;
  ListBase *nubase = &editnurb->nurbs;
  const int nuindex = get_nurb_index(nubase, nu);
  if (cu->actnu == nuindex) {
    cu->actnu = CU_ACT_NONE;
  }

  BLI_remlink(nubase, nu);
  BKE_nurb_free(nu);
  nu = NULL;
}

static void delete_bezt_from_nurb(const BezTriple *bezt, Nurb *nu)
{
  BLI_assert(nu->type == CU_BEZIER);
  const int index = BKE_curve_nurb_vert_index_get(nu, bezt);
  nu->pntsu -= 1;
  memmove(nu->bezt + index, nu->bezt + index + 1, (nu->pntsu - index) * sizeof(BezTriple));
}

static void delete_bp_from_nurb(const BPoint *bp, Nurb *nu)
{
  BLI_assert(nu->type == CU_NURBS || nu->type == CU_POLY);
  const int index = BKE_curve_nurb_vert_index_get(nu, bp);
  nu->pntsu -= 1;
  memmove(nu->bp + index, nu->bp + index + 1, (nu->pntsu - index) * sizeof(BPoint));
}

/* Get the closest point on an edge to a given point based on perpendicular distance. Return true
 * if the closest point falls on the edge.  */
static bool get_closest_point_on_edge(float r_point[3],
                                      const float pos[2],
                                      const float pos1[3],
                                      const float pos2[3],
                                      const ViewContext *vc,
                                      float *r_factor)
{
  float pos1_2d[2], pos2_2d[2], vec1[2], vec2[2], vec3[2];

  /* Get screen space coordinates of points. */
  if (!(ED_view3d_project_float_object(
            vc->region, pos1, pos1_2d, V3D_PROJ_RET_CLIP_BB | V3D_PROJ_RET_CLIP_WIN) ==
            V3D_PROJ_RET_OK &&
        ED_view3d_project_float_object(
            vc->region, pos2, pos2_2d, V3D_PROJ_RET_CLIP_BB | V3D_PROJ_RET_CLIP_WIN) ==
            V3D_PROJ_RET_OK)) {
    return false;
  }

  /* Obtain the vectors of each side. */
  sub_v2_v2v2(vec1, pos, pos1_2d);
  sub_v2_v2v2(vec2, pos2_2d, pos);
  sub_v2_v2v2(vec3, pos2_2d, pos1_2d);

  const float dot1 = dot_v2v2(vec1, vec3);
  const float dot2 = dot_v2v2(vec2, vec3);

  /* Compare the dot products to identify if both angles are optuse/acute or
  opposite to each other. If they're the same, that indicates that there is a
  perpendicular line from the mouse to the line.*/
  if ((dot1 > 0) == (dot2 > 0)) {
    float len_vec3_sq = len_squared_v2(vec3);
    *r_factor = 1 - dot2 / len_vec3_sq;

    float pos_dif[3];
    sub_v3_v3v3(pos_dif, pos2, pos1);
    madd_v3_v3v3fl(r_point, pos1, pos_dif, *r_factor);
    return true;
  }

  if (len_manhattan_v2(vec1) < len_manhattan_v2(vec2)) {
    copy_v3_v3(r_point, pos1);
    return false;
  }
  copy_v3_v3(r_point, pos2);
  return false;
}

/* Get closest vertex in all nurbs in given #ListBase to a given point. */
static void get_closest_vertex_to_point_in_nurbs(ListBase *nurbs,
                                                 Nurb **r_nu,
                                                 BezTriple **r_bezt,
                                                 BPoint **r_bp,
                                                 const float point[2],
                                                 const ViewContext *vc)
{
  float min_distance_bezt = FLT_MAX;
  float min_distance_bp = FLT_MAX;

  BezTriple *closest_bezt = NULL;
  BPoint *closest_bp = NULL;
  Nurb *closest_bezt_nu = NULL;
  Nurb *closest_bp_nu = NULL;

  LISTBASE_FOREACH (Nurb *, nu, nurbs) {
    if (nu->type == CU_BEZIER) {
      for (int i = 0; i < nu->pntsu; i++) {
        BezTriple *bezt = &nu->bezt[i];
        float bezt_vec[2];
        if (ED_view3d_project_float_object(vc->region,
                                           bezt->vec[1],
                                           bezt_vec,
                                           V3D_PROJ_TEST_CLIP_BB | V3D_PROJ_TEST_CLIP_WIN) ==
            V3D_PROJ_RET_OK) {
          const float distance = len_manhattan_v2v2(bezt_vec, point);
          if (distance < min_distance_bezt) {
            min_distance_bezt = distance;
            closest_bezt = bezt;
            closest_bezt_nu = nu;
          }
        }
      }
    }
    else {
      for (int i = 0; i < nu->pntsu; i++) {
        BPoint *bp = &nu->bp[i];
        float bp_vec[2];
        if (ED_view3d_project_float_object(
                vc->region, bp->vec, bp_vec, V3D_PROJ_TEST_CLIP_BB | V3D_PROJ_TEST_CLIP_WIN) ==
            V3D_PROJ_RET_OK) {
          const float distance = len_manhattan_v2v2(bp_vec, point);
          if (distance < min_distance_bp) {
            min_distance_bp = distance;
            closest_bp = bp;
            closest_bp_nu = nu;
          }
        }
      }
    }
  }

  const float threshold_distance = ED_view3d_select_dist_px();
  if (min_distance_bezt < threshold_distance || min_distance_bp < threshold_distance) {
    if (min_distance_bp < min_distance_bezt) {
      *r_bp = closest_bp;
      *r_nu = closest_bp_nu;
    }
    else {
      *r_bezt = closest_bezt;
      *r_nu = closest_bezt_nu;
    }
  }
}

/* Assign values for several frequently changing attributes of #CutData. */
static void assign_cut_data(CutData *data,
                            const float min_dist,
                            Nurb *nu,
                            const int bext_index,
                            const int bp_index,
                            const float parameter,
                            const float cut_loc[3])
{
  data->min_dist = min_dist;
  data->nurb = nu;
  data->bezt_index = bext_index;
  data->bp_index = bp_index;
  data->parameter = parameter;
  copy_v3_v3(data->cut_loc, cut_loc);
}

/* Iterate over all the geometry between the segment formed by bezt1 and bezt2
 * to find the closest edge to #data->mval (mouse location) and update #data->prev_loc
 * and #data->next_loc with the vertices of the edge. */
static void update_data_if_closest_bezt_in_segment(const BezTriple *bezt1,
                                                   const BezTriple *bezt2,
                                                   Nurb *nu,
                                                   const int index,
                                                   const ViewContext *vc,
                                                   CutData *data)
{
  const float resolu = nu->resolu;
  float *points = MEM_mallocN(sizeof(float[3]) * (resolu + 1), __func__);

  /* Calculate all points on curve. */
  for (int j = 0; j < 3; j++) {
    BKE_curve_forward_diff_bezier(bezt1->vec[1][j],
                                  bezt1->vec[2][j],
                                  bezt2->vec[0][j],
                                  bezt2->vec[1][j],
                                  points + j,
                                  resolu,
                                  sizeof(float[3]));
  }

  for (int k = 0; k <= resolu; k++) {
    float screen_co[2];

    bool check = ED_view3d_project_float_object(vc->region,
                                                points + 3 * k,
                                                screen_co,
                                                V3D_PROJ_RET_CLIP_BB | V3D_PROJ_RET_CLIP_WIN) ==
                 V3D_PROJ_RET_OK;

    if (check) {
      const float distance = len_manhattan_v2v2(screen_co, data->mval);
      if (distance < data->min_dist) {
        assign_cut_data(data, distance, nu, index, -1, ((float)k) / resolu, points + 3 * k);

        data->has_prev = k > 0;
        data->has_next = k < resolu;
        if (data->has_prev) {
          copy_v3_v3(data->prev_loc, points + 3 * (k - 1));
        }
        if (data->has_next) {
          copy_v3_v3(data->next_loc, points + 3 * (k + 1));
        }
      }
    }
  }
  MEM_freeN(points);
}

/* Interpolate along the Bezier segment by a parameter (between 0 and 1) and get its location. */
static void get_bezier_interpolated_point(float r_point[3],
                                          const BezTriple *bezt1,
                                          const BezTriple *bezt2,
                                          const float parameter)
{
  float tmp1[3], tmp2[3], tmp3[3];
  interp_v3_v3v3(tmp1, bezt1->vec[1], bezt1->vec[2], parameter);
  interp_v3_v3v3(tmp2, bezt1->vec[2], bezt2->vec[0], parameter);
  interp_v3_v3v3(tmp3, bezt2->vec[0], bezt2->vec[1], parameter);
  interp_v3_v3v3(tmp1, tmp1, tmp2, parameter);
  interp_v3_v3v3(tmp2, tmp2, tmp3, parameter);
  interp_v3_v3v3(r_point, tmp1, tmp2, parameter);
}

/* Update the closest location as cut location in data. */
static void update_cut_loc_in_data(CutData *data, const ViewContext *vc)
{
  bool found_min = false;
  float point[3];
  float factor;

  if (data->has_prev) {
    found_min = get_closest_point_on_edge(
        point, data->mval, data->cut_loc, data->prev_loc, vc, &factor);
    factor = -factor;
  }
  if (!found_min && data->has_next) {
    found_min = get_closest_point_on_edge(
        point, data->mval, data->cut_loc, data->next_loc, vc, &factor);
  }
  if (found_min) {
    float point_2d[2];
    if (ED_view3d_project_float_object(
            vc->region, point, point_2d, V3D_PROJ_TEST_CLIP_BB | V3D_PROJ_TEST_CLIP_WIN) !=
        V3D_PROJ_RET_OK) {
      return;
    }
    const float dist = len_manhattan_v2v2(point_2d, data->mval);
    data->min_dist = dist;
    data->parameter += factor / data->nurb->resolu;

    Nurb *nu = data->nurb;
    get_bezier_interpolated_point(data->cut_loc,
                                  &nu->bezt[data->bezt_index],
                                  &nu->bezt[(data->bezt_index + 1) % (nu->pntsu)],
                                  data->parameter);
  }
}

/* Calculate handle positions of added and adjacent control points such that shape is preserved. */
static void calculate_new_bezier_point(const float point_prev[3],
                                       float handle_prev[3],
                                       float new_left_handle[3],
                                       float new_right_handle[3],
                                       float handle_next[3],
                                       const float point_next[3],
                                       const float parameter)
{
  float center_point[3];
  interp_v3_v3v3(center_point, handle_prev, handle_next, parameter);
  interp_v3_v3v3(handle_prev, point_prev, handle_prev, parameter);
  interp_v3_v3v3(handle_next, handle_next, point_next, parameter);
  interp_v3_v3v3(new_left_handle, handle_prev, center_point, parameter);
  interp_v3_v3v3(new_right_handle, center_point, handle_next, parameter);
}

/* Update the nearest point data for all nurbs. */
static void update_data_for_all_nurbs(const ListBase *nurbs, const ViewContext *vc, CutData *data)
{
  LISTBASE_FOREACH (Nurb *, nu, nurbs) {
    if (nu->type == CU_BEZIER) {
      float screen_co[2];
      if (data->nurb == NULL) {
        ED_view3d_project_float_object(vc->region,
                                       nu->bezt->vec[1],
                                       screen_co,
                                       V3D_PROJ_TEST_CLIP_BB | V3D_PROJ_TEST_CLIP_WIN);
        assign_cut_data(data, FLT_MAX, nu, 0, -1, 0.0f, nu->bezt->vec[1]);
      }

      BezTriple *bezt = NULL;
      for (int i = 0; i < nu->pntsu - 1; i++) {
        bezt = &nu->bezt[i];
        update_data_if_closest_bezt_in_segment(bezt, bezt + 1, nu, i, vc, data);
      }

      if (nu->flagu & CU_NURB_CYCLIC && bezt) {
        update_data_if_closest_bezt_in_segment(bezt + 1, nu->bezt, nu, nu->pntsu - 1, vc, data);
      }
    }
    else {
      float screen_co[2];
      if (data->nurb == NULL) {
        ED_view3d_project_float_object(
            vc->region, nu->bp->vec, screen_co, V3D_PROJ_TEST_CLIP_BB | V3D_PROJ_TEST_CLIP_WIN);
        assign_cut_data(
            data, len_manhattan_v2v2(screen_co, data->mval), nu, -1, 0, 0.0f, nu->bp->vec);
      }

      for (int i = 0; i < nu->pntsu - 1; i++) {
        float point[3], factor;
        bool found_min = get_closest_point_on_edge(
            point, data->mval, (nu->bp + i)->vec, (nu->bp + i + 1)->vec, vc, &factor);
        if (found_min) {
          float point_2d[2];
          if (ED_view3d_project_float_object(
                  vc->region, point, point_2d, V3D_PROJ_TEST_CLIP_BB | V3D_PROJ_TEST_CLIP_WIN) ==
              V3D_PROJ_RET_OK) {
            const float dist = len_manhattan_v2v2(point_2d, data->mval);
            if (dist < data->min_dist) {
              assign_cut_data(data, dist, nu, -1, i, 0.0f, point);
            }
          }
        }
      }
    }
  }
}

/* Insert a #BezTriple to a nurb at the location specified by `op_data`. */
static void add_bezt_to_nurb(Nurb *nu, const CutData *data, Curve *cu)
{
  EditNurb *editnurb = cu->editnurb;

  BezTriple *bezt1 = (BezTriple *)MEM_mallocN((nu->pntsu + 1) * sizeof(BezTriple), __func__);
  const int index = data->bezt_index + 1;
  /* Copy all control points before the cut to the new memory. */
  memcpy(bezt1, nu->bezt, index * sizeof(BezTriple));
  BezTriple *new_bezt = bezt1 + index;

  /* Duplicate control point after the cut. */
  memcpy(new_bezt, new_bezt - 1, sizeof(BezTriple));
  copy_v3_v3(new_bezt->vec[1], data->cut_loc);

  if (index < nu->pntsu) {
    /* Copy all control points after the cut to the new memory. */
    memcpy(bezt1 + index + 1, nu->bezt + index, (nu->pntsu - index) * sizeof(BezTriple));
  }

  nu->pntsu += 1;
  cu->actvert = CU_ACT_NONE;

  BezTriple *next_bezt;
  if ((nu->flagu & CU_NURB_CYCLIC) && (index == nu->pntsu - 1)) {
    next_bezt = bezt1;
  }
  else {
    next_bezt = new_bezt + 1;
  }

  /* Interpolate radius, tilt, weight */
  new_bezt->tilt = interpf(next_bezt->tilt, (new_bezt - 1)->tilt, data->parameter);
  new_bezt->radius = interpf(next_bezt->radius, (new_bezt - 1)->radius, data->parameter);
  new_bezt->weight = interpf(next_bezt->weight, (new_bezt - 1)->weight, data->parameter);

  new_bezt->h1 = new_bezt->h2 = HD_ALIGN;

  calculate_new_bezier_point((new_bezt - 1)->vec[1],
                             (new_bezt - 1)->vec[2],
                             new_bezt->vec[0],
                             new_bezt->vec[2],
                             next_bezt->vec[0],
                             next_bezt->vec[1],
                             data->parameter);

  MEM_freeN(nu->bezt);
  nu->bezt = bezt1;
  ED_curve_deselect_all(editnurb);
  BKE_nurb_handles_calc(nu);
  BEZT_SEL_ALL(new_bezt);
}

/* Insert a #BPoint to a nurb at the location specified by `op_data`. */
static void add_bp_to_nurb(Nurb *nu, const CutData *data, Curve *cu)
{
  EditNurb *editnurb = cu->editnurb;

  BPoint *bp1 = (BPoint *)MEM_mallocN((nu->pntsu + 1) * sizeof(BPoint), __func__);
  const int index = data->bp_index + 1;
  /* Copy all control points before the cut to the new memory. */
  memcpy(bp1, nu->bp, index * sizeof(BPoint));
  BPoint *new_bp = bp1 + index;

  /* Duplicate control point after the cut. */
  memcpy(new_bp, new_bp - 1, sizeof(BPoint));
  copy_v3_v3(new_bp->vec, data->cut_loc);

  if (index < nu->pntsu) {
    /* Copy all control points after the cut to the new memory. */
    memcpy(bp1 + index + 1, nu->bp + index, (nu->pntsu - index) * sizeof(BPoint));
  }

  nu->pntsu += 1;
  cu->actvert = CU_ACT_NONE;

  BPoint *next_bp;
  if ((nu->flagu & CU_NURB_CYCLIC) && (index == nu->pntsu - 1)) {
    next_bp = bp1;
  }
  else {
    next_bp = new_bp + 1;
  }

  /* Interpolate radius, tilt, weight */
  new_bp->tilt = interpf(next_bp->tilt, (new_bp - 1)->tilt, data->parameter);
  new_bp->radius = interpf(next_bp->radius, (new_bp - 1)->radius, data->parameter);
  new_bp->weight = interpf(next_bp->weight, (new_bp - 1)->weight, data->parameter);

  MEM_freeN(nu->bp);
  nu->bp = bp1;
  ED_curve_deselect_all(editnurb);
  BKE_nurb_knot_calc_u(nu);
}

/* Make a cut on the nearest nurb at the closest point. */
static void make_cut(const wmEvent *event, Curve *cu, Nurb **r_nu, const ViewContext *vc)
{
  CutData data = {.bezt_index = 0,
                  .bp_index = 0,
                  .min_dist = FLT_MAX,
                  .parameter = 0.5f,
                  .has_prev = false,
                  .has_next = false,
                  .mval[0] = event->mval[0],
                  .mval[1] = event->mval[1]};

  ListBase *nurbs = BKE_curve_editNurbs_get(cu);

  update_data_for_all_nurbs(nurbs, vc, &data);

  const float threshold_distance = ED_view3d_select_dist_px();
  Nurb *nu = data.nurb;
  if (nu) {
    if (nu->type == CU_BEZIER) {
      update_cut_loc_in_data(&data, vc);
      if (data.min_dist < threshold_distance) {
        add_bezt_to_nurb(nu, &data, cu);
        *r_nu = nu;
      }
    }
    else if (data.min_dist < threshold_distance) {
      add_bp_to_nurb(nu, &data, cu);
    }
  }
}

/* Add a new vertex connected to the selected vertex. */
static void add_vertex_connected_to_selected_vertex(const ViewContext *vc,
                                                    Object *obedit,
                                                    const wmEvent *event)
{
  Nurb *nu = NULL;
  BezTriple *bezt = NULL;
  BPoint *bp = NULL;
  Curve *cu = vc->obedit->data;

  float location[3];

  ED_curve_nurb_vert_selected_find(cu, vc->v3d, &nu, &bezt, &bp);

  if (bezt) {
    mul_v3_m4v3(location, vc->obedit->obmat, bezt->vec[1]);
  }
  else if (bp) {
    mul_v3_m4v3(location, vc->obedit->obmat, bp->vec);
  }
  else {
    copy_v3_v3(location, vc->scene->cursor.location);
  }

  ED_view3d_win_to_3d_int(vc->v3d, vc->region, location, event->mval, location);
  EditNurb *editnurb = cu->editnurb;

  float imat[4][4];
  invert_m4_m4(imat, obedit->obmat);
  mul_m4_v3(imat, location);

  Nurb *old_last_nu = editnurb->nurbs.last;
  ed_editcurve_addvert(cu, editnurb, vc->v3d, location);
  Nurb *new_last_nu = editnurb->nurbs.last;

  if (old_last_nu != new_last_nu) {
    new_last_nu->flagu = ~CU_NURB_CYCLIC;
  }

  ED_curve_nurb_vert_selected_find(cu, vc->v3d, &nu, &bezt, &bp);
  if (bezt) {
    bezt->h1 = HD_VECT;
    bezt->h2 = HD_VECT;
  }
}

/* Check if a spline segment is nearby. */
static bool is_spline_nearby(ViewContext *vc, wmOperator *op, const wmEvent *event)
{
  Curve *cu = vc->obedit->data;
  ListBase *nurbs = BKE_curve_editNurbs_get(cu);

  CutData data = {.bezt_index = 0,
                  .bp_index = 0,
                  .min_dist = FLT_MAX,
                  .parameter = 0.5f,
                  .has_prev = false,
                  .has_next = false,
                  .mval[0] = event->mval[0],
                  .mval[1] = event->mval[1]};

  update_data_for_all_nurbs(nurbs, vc, &data);

  const float threshold_distance = ED_view3d_select_dist_px();
  if (data.nurb && !data.nurb->bp && data.min_dist < threshold_distance) {
    MoveSegmentData *seg_data;
    op->customdata = seg_data = MEM_callocN(sizeof(MoveSegmentData), __func__);
    seg_data->bezt_index = data.bezt_index;
    seg_data->nu = data.nurb;
    return true;
  }
  return false;
}

/* Move segment to mouse pointer. */
static void move_segment(MoveSegmentData *seg_data, const wmEvent *event, ViewContext *vc)
{
  Nurb *nu = seg_data->nu;
  BezTriple *bezt1 = nu->bezt + seg_data->bezt_index;
  BezTriple *bezt2 = BKE_nurb_bezt_get_next(nu, bezt1);

  float mouse_3d[3];
  float depth[3];
  /* Use the center of the spline segment as depth. */
  get_bezier_interpolated_point(depth, bezt1, bezt2, 0.5f);
  mouse_location_to_worldspace(event->mval, depth, vc, mouse_3d);

  /*
   * Equation of Bezier Curve
   *      => B(t) = (1-t)^3 * P0 + 3(1-t)^2 * t * P1 + 3(1-t) * t^2 * P2 + t^3 * P3
   * Mouse location (Say Pm) should satisfy this equation.
   * Substituting t = 0.5 => Pm = 0.5^3 * (P0 + 3P1 + 3P2 + P3)
   * Therefore => P1 + P2 = (8 * Pm - P0 - P3) / 3
   *
   * Another constraint is required to identify P1 and P2.
   * The constraint is to minimize the distance between new points and initial points.
   * The minima can be found by differentiating the total distance.
   */

  float p1_plus_p2_div_2[3];
  p1_plus_p2_div_2[0] = (8.0f * mouse_3d[0] - bezt1->vec[1][0] - bezt2->vec[1][0]) / 6.0f;
  p1_plus_p2_div_2[1] = (8.0f * mouse_3d[1] - bezt1->vec[1][1] - bezt2->vec[1][1]) / 6.0f;
  p1_plus_p2_div_2[2] = (8.0f * mouse_3d[2] - bezt1->vec[1][2] - bezt2->vec[1][2]) / 6.0f;

  float p1_minus_p2_div_2[3];
  sub_v3_v3v3(p1_minus_p2_div_2, bezt1->vec[2], bezt2->vec[0]);
  mul_v3_fl(p1_minus_p2_div_2, 0.5f);

  add_v3_v3v3(bezt1->vec[2], p1_plus_p2_div_2, p1_minus_p2_div_2);
  sub_v3_v3v3(bezt2->vec[0], p1_plus_p2_div_2, p1_minus_p2_div_2);

  free_up_handles_for_movement(bezt1, true, true);
  free_up_handles_for_movement(bezt2, true, true);

  /* Move opposite handle as well if type is align. */
  if (bezt1->h1 == HD_ALIGN) {
    float handle_vec[3];
    sub_v3_v3v3(handle_vec, bezt1->vec[1], bezt1->vec[2]);
    normalize_v3_length(handle_vec, len_v3v3(bezt1->vec[1], bezt1->vec[0]));
    add_v3_v3v3(bezt1->vec[0], bezt1->vec[1], handle_vec);
  }
  if (bezt2->h2 == HD_ALIGN) {
    float handle_vec[3];
    sub_v3_v3v3(handle_vec, bezt2->vec[1], bezt2->vec[0]);
    normalize_v3_length(handle_vec, len_v3v3(bezt2->vec[1], bezt2->vec[2]));
    add_v3_v3v3(bezt2->vec[2], bezt2->vec[1], handle_vec);
  }
}

/* Close the spline if endpoints are selected consecutively. Return true if cycle was created. */
static bool make_cyclic_if_endpoints(
    Nurb *sel_nu, BezTriple *sel_bezt, BPoint *sel_bp, ViewContext *vc, bContext *C)
{
  if (sel_bezt || sel_bp) {
    const bool is_bezt_endpoint = (sel_nu->type == CU_BEZIER &&
                                   (sel_bezt == sel_nu->bezt ||
                                    sel_bezt == sel_nu->bezt + sel_nu->pntsu - 1));
    const bool is_bp_endpoint = (sel_nu->type != CU_BEZIER &&
                                 (sel_bp == sel_nu->bp ||
                                  sel_bp == sel_nu->bp + sel_nu->pntsu - 1));
    if (!(is_bezt_endpoint || is_bp_endpoint)) {
      return false;
    }

    short hand;
    Nurb *nu = NULL;
    BezTriple *bezt = NULL;
    BPoint *bp = NULL;
    Base *basact = NULL;
    ED_curve_pick_vert(vc, 1, &nu, &bezt, &bp, &hand, &basact);

    if (nu == sel_nu && ((nu->type == CU_BEZIER && bezt != sel_bezt &&
                          (bezt == nu->bezt || bezt == nu->bezt + nu->pntsu - 1)) ||
                         (nu->type != CU_BEZIER && bp != sel_bp &&
                          (bp == nu->bp || bp == nu->bp + nu->pntsu - 1)))) {
      View3D *v3d = CTX_wm_view3d(C);
      ListBase *editnurb = object_editcurve_get(vc->obedit);
      ed_curve_toggle_cyclic(v3d, editnurb, 0);
      return true;
    }
  }
  return false;
}

enum {
  PEN_MODAL_FREE_MOVE_HANDLE = 1,
};

wmKeyMap *curve_pen_modal_keymap(wmKeyConfig *keyconf)
{
  static const EnumPropertyItem modal_items[] = {
      {PEN_MODAL_FREE_MOVE_HANDLE,
       "FREE_MOVE_HANDLE",
       0,
       "Free Move handle",
       "Move handle of newly added point freely"},
      {0, NULL, 0, NULL, NULL},
  };

  wmKeyMap *keymap = WM_modalkeymap_find(keyconf, "Curve Pen Modal Map");

  if (keymap && keymap->modal_items) {
    return NULL;
  }

  keymap = WM_modalkeymap_ensure(keyconf, "Curve Pen Modal Map", modal_items);

  WM_modalkeymap_assign(keymap, "CURVE_OT_pen");

  return keymap;
}

static int curve_pen_modal(bContext *C, wmOperator *op, const wmEvent *event)
{
  Depsgraph *depsgraph = CTX_data_ensure_evaluated_depsgraph(C);
  ViewContext vc;
  Object *obedit = CTX_data_edit_object(C);

  ED_view3d_viewcontext_init(C, &vc, depsgraph);

  BezTriple *bezt = NULL;
  BPoint *bp = NULL;
  Nurb *nu = NULL;

  int ret = OPERATOR_RUNNING_MODAL;
  /* Whether the mouse is clicking and dragging. */
  bool dragging = RNA_boolean_get(op->ptr, "dragging");
  /* Whether a new point was added at the beginning of tool execution. */
  const bool is_new_point = RNA_boolean_get(op->ptr, "new");
  /* Whether a segment is being altered by click and drag. */
  bool moving_segment = RNA_boolean_get(op->ptr, "moving_segment");

  if (event->type == EVT_MODAL_MAP) {
    if (event->val == PEN_MODAL_FREE_MOVE_HANDLE) {
      ED_curve_nurb_vert_selected_find(vc.obedit->data, vc.v3d, &nu, &bezt, &bp);

      if (bezt) {
        if (bezt->h1 != HD_FREE || bezt->h2 != HD_FREE) {
          bezt->h1 = bezt->h2 = HD_FREE;
        }
        else {
          bezt->h1 = bezt->h2 = HD_ALIGN;
          BKE_nurb_handles_calc(nu);
        }
        RNA_boolean_set(op->ptr, "dragging", true);
        dragging = true;
      }
    }
  }

  if (ELEM(event->type, MOUSEMOVE, INBETWEEN_MOUSEMOVE)) {
    if (!dragging && WM_event_drag_test(event, event->prev_click_xy) && event->val == KM_PRESS) {
      RNA_boolean_set(op->ptr, "dragging", true);
      dragging = true;
    }
    if (dragging) {
      if (moving_segment) {
        MoveSegmentData *seg_data = op->customdata;
        nu = seg_data->nu;
        move_segment(seg_data, event, &vc);
      }
      /* If dragging a new control point, move handle point with mouse cursor. Else move entire
       * control point. */
      else if (is_new_point) {
        ED_curve_nurb_vert_selected_find(vc.obedit->data, vc.v3d, &nu, &bezt, &bp);
        if (bezt) {
          /* Move opposite handle if last vertex. */
          const bool invert = (nu->bezt + nu->pntsu - 1 == bezt &&
                               !(nu->flagu & CU_NURB_CYCLIC)) ||
                              (nu->bezt == bezt && (nu->flagu & CU_NURB_CYCLIC));
          move_bezt_handles_to_mouse(bezt, invert, event, &vc);
        }
      }
      else {
        ED_curve_nurb_vert_selected_find(vc.obedit->data, vc.v3d, &nu, &bezt, &bp);
        if (bezt) {
          move_selected_bezt_to_mouse(bezt, &vc, event);
        }
        else if (bp) {
          move_bp_to_mouse(bp, event, &vc);
        }
      }
      if (nu && nu->type == CU_BEZIER) {
        BKE_nurb_handles_calc(nu);
      }
    }
  }
  else if (ELEM(event->type, LEFTMOUSE)) {
    if (event->val == KM_PRESS) {
      Curve *cu = vc.obedit->data;
      /* Get currently selected point if any. Used for making spline cyclic. */
      ED_curve_nurb_vert_selected_find(cu, vc.v3d, &nu, &bezt, &bp);

      const bool found_point = ED_curve_editnurb_select_pick(C, event->mval, false, false, false);
      RNA_boolean_set(op->ptr, "new", !found_point);

      if (found_point) {
        copy_v2_v2_int(vc.mval, event->mval);
        if (nu && !(nu->flagu & CU_NURB_CYCLIC)) {
          const bool closed = nu->pntsu > 2 && make_cyclic_if_endpoints(nu, bezt, bp, &vc, C);

          /* Set "new" to true to be able to click and drag to control handles when added. */
          RNA_boolean_set(op->ptr, "new", closed);
        }
      }
      else {
        if (is_spline_nearby(&vc, op, event)) {
          RNA_boolean_set(op->ptr, "moving_segment", true);
          moving_segment = true;
        }
        else {
          add_vertex_connected_to_selected_vertex(&vc, obedit, event);
        }
      }
    }
    else if (event->val == KM_RELEASE) {
      if (moving_segment) {
        if (!dragging) {
          add_vertex_connected_to_selected_vertex(&vc, obedit, event);
        }
        else {
          MEM_freeN(op->customdata);
        }
      }
      RNA_boolean_set(op->ptr, "dragging", false);
      RNA_boolean_set(op->ptr, "new", false);
      RNA_boolean_set(op->ptr, "moving_segment", false);
      ret = OPERATOR_FINISHED;
    }
  }

  WM_event_add_notifier(C, NC_GEOM | ND_DATA, obedit->data);
  WM_event_add_notifier(C, NC_GEOM | ND_SELECT, obedit->data);
  DEG_id_tag_update(obedit->data, 0);

  return ret;
}

static int curve_pen_delete_modal(bContext *C, wmOperator *op, const wmEvent *event)
{
  Depsgraph *depsgraph = CTX_data_ensure_evaluated_depsgraph(C);
  ViewContext vc;
  Object *obedit = CTX_data_edit_object(C);

  ED_view3d_viewcontext_init(C, &vc, depsgraph);

  BezTriple *bezt = NULL;
  BPoint *bp = NULL;
  Nurb *nu = NULL;

  int ret = OPERATOR_RUNNING_MODAL;

  if (ELEM(event->type, LEFTMOUSE)) {
    if (event->val == KM_PRESS) {
      Curve *cu = vc.obedit->data;
      ListBase *nurbs = BKE_curve_editNurbs_get(cu);
      float mouse_point[2] = {(float)event->mval[0], (float)event->mval[1]};

      get_closest_vertex_to_point_in_nurbs(nurbs, &nu, &bezt, &bp, mouse_point, &vc);
      const bool found_point = nu != NULL;

      if (found_point) {
        ED_curve_deselect_all(cu->editnurb);
        if (nu) {
          if (nu->type == CU_BEZIER) {
            BezTriple *next_bezt = BKE_nurb_bezt_get_next(nu, bezt);
            BezTriple *prev_bezt = BKE_nurb_bezt_get_prev(nu, bezt);
            if (next_bezt && prev_bezt) {
              const int bez_index = BKE_curve_nurb_vert_index_get(nu, bezt);
              uint span_step[2] = {bez_index, bez_index};
              ed_dissolve_bez_segment(prev_bezt, next_bezt, nu, cu, 1, span_step);
            }
            delete_bezt_from_nurb(bezt, nu);
          }
          else {
            delete_bp_from_nurb(bp, nu);
          }

          if (nu->pntsu == 0) {
            delete_nurb(cu, nu);
          }
        }
      }

      if (nu && nu->type == CU_BEZIER) {
        BKE_nurb_handles_calc(nu);
      }
    }
    else if (event->val == KM_RELEASE) {
      ret = OPERATOR_FINISHED;
    }
  }

  WM_event_add_notifier(C, NC_GEOM | ND_DATA, obedit->data);
  WM_event_add_notifier(C, NC_GEOM | ND_SELECT, obedit->data);
  DEG_id_tag_update(obedit->data, 0);

  return ret;
}

static int curve_pen_insert_modal(bContext *C, wmOperator *op, const wmEvent *event)
{
  Depsgraph *depsgraph = CTX_data_ensure_evaluated_depsgraph(C);
  ViewContext vc;
  Object *obedit = CTX_data_edit_object(C);

  ED_view3d_viewcontext_init(C, &vc, depsgraph);

  BezTriple *bezt = NULL;
  BPoint *bp = NULL;
  Nurb *nu = NULL;

  int ret = OPERATOR_RUNNING_MODAL;

  if (ELEM(event->type, LEFTMOUSE)) {
    if (event->val == KM_PRESS) {
      Curve *cu = vc.obedit->data;
      ListBase *nurbs = BKE_curve_editNurbs_get(cu);
      float mouse_point[2] = {(float)event->mval[0], (float)event->mval[1]};

      get_closest_vertex_to_point_in_nurbs(nurbs, &nu, &bezt, &bp, mouse_point, &vc);
      const bool no_point = nu == NULL;

      make_cut(event, cu, &nu, &vc);

      if (nu && nu->type == CU_BEZIER) {
        BKE_nurb_handles_calc(nu);
      }
    }
    else if (event->val == KM_RELEASE) {
      ret = OPERATOR_FINISHED;
    }
  }

  WM_event_add_notifier(C, NC_GEOM | ND_DATA, obedit->data);
  WM_event_add_notifier(C, NC_GEOM | ND_SELECT, obedit->data);
  DEG_id_tag_update(obedit->data, 0);

  return ret;
}

static int curve_pen_invoke(bContext *C, wmOperator *op, const wmEvent *event)
{
  int ret = curve_pen_modal(C, op, event);
  BLI_assert(ret == OPERATOR_RUNNING_MODAL);
  if (ret == OPERATOR_RUNNING_MODAL) {
    WM_event_add_modal_handler(C, op);
  }

  return ret;
}

static int curve_pen_delete_invoke(bContext *C, wmOperator *op, const wmEvent *event)
{
  int ret = curve_pen_delete_modal(C, op, event);
  BLI_assert(ret == OPERATOR_RUNNING_MODAL);
  if (ret == OPERATOR_RUNNING_MODAL) {
    WM_event_add_modal_handler(C, op);
  }

  return ret;
}

static int curve_pen_insert_invoke(bContext *C, wmOperator *op, const wmEvent *event)
{
  int ret = curve_pen_insert_modal(C, op, event);
  BLI_assert(ret == OPERATOR_RUNNING_MODAL);
  if (ret == OPERATOR_RUNNING_MODAL) {
    WM_event_add_modal_handler(C, op);
  }

  return ret;
}

void CURVE_OT_pen(wmOperatorType *ot)
{
  /* identifiers */
  ot->name = "Curve Pen";
  ot->idname = "CURVE_OT_pen";
  ot->description = "Construct and edit splines";

  /* api callbacks */
  ot->invoke = curve_pen_invoke;
  ot->modal = curve_pen_modal;
  ot->poll = ED_operator_view3d_active;

  /* flags */
  ot->flag = OPTYPE_UNDO;

  /* properties */
  WM_operator_properties_mouse_select(ot);

  PropertyRNA *prop;
  prop = RNA_def_boolean(ot->srna, "dragging", 0, "Dragging", "Check if click and drag");
  RNA_def_property_flag(prop, PROP_HIDDEN | PROP_SKIP_SAVE);
  prop = RNA_def_boolean(
      ot->srna, "new", 0, "New Point Drag", "The point was added with the press before drag");
  RNA_def_property_flag(prop, PROP_HIDDEN | PROP_SKIP_SAVE);
  prop = RNA_def_boolean(ot->srna, "moving_segment", 0, "Moving Segment", "");
  RNA_def_property_flag(prop, PROP_HIDDEN | PROP_SKIP_SAVE);
}

void CURVE_OT_pen_delete(wmOperatorType *ot)
{
  /* identifiers */
  ot->name = "Curve Pen Delete";
  ot->idname = "CURVE_OT_pen_delete";
  ot->description = "Delete control points";

  /* api callbacks */
  ot->invoke = curve_pen_delete_invoke;
  ot->modal = curve_pen_delete_modal;
  ot->poll = ED_operator_view3d_active;

  /* flags */
  ot->flag = OPTYPE_UNDO;

  /* properties */
  WM_operator_properties_mouse_select(ot);
}

void CURVE_OT_pen_insert(wmOperatorType *ot)
{
  /* identifiers */
  ot->name = "Curve Pen Insert";
  ot->idname = "CURVE_OT_pen_insert";
  ot->description = "Insert control points into segments";

  /* api callbacks */
  ot->invoke = curve_pen_insert_invoke;
  ot->modal = curve_pen_insert_modal;
  ot->poll = ED_operator_view3d_active;

  /* flags */
  ot->flag = OPTYPE_UNDO;

  /* properties */
  WM_operator_properties_mouse_select(ot);
}
