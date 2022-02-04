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
#include "DNA_windowmanager_types.h"

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

#define FOREACH_SELECTED_BEZT_BEGIN(bezt, nurbs) \
  LISTBASE_FOREACH (Nurb *, nu, nurbs) { \
    if (nu->type == CU_BEZIER) { \
      for (int i = 0; i < nu->pntsu; i++) { \
        BezTriple *bezt = nu->bezt + i; \
        if (BEZT_ISSEL_ANY(bezt)) {

#define FOREACH_SELECTED_BEZT_END \
  } \
  } \
  } \
  BKE_nurb_handles_calc(nu); \
  }

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
  /* Fraction along the segment at which mouse was pressed. */
  float t;
} MoveSegmentData;

typedef struct CurvePenData {
  MoveSegmentData *msd;
  /* Whether the mouse is clicking and dragging. */
  bool dragging;
  /* Whether a new point was added at the beginning of tool execution. */
  bool new_point;
  /* Whether a segment is being altered by click and drag. */
  bool spline_nearby;
  /* Whether shortcut for toggling free handles was pressed. */
  bool free_toggle_pressed;
  /* Whether shortcut for linking handles was pressed. */
  bool link_handles_pressed;
  /* Whether the current state of the moved handle is linked. */
  bool link_handles;
  /* Whether the current state of the handle angle is locked. */
  bool lock_angle;
  /* Whether some action was done. Used for select. */
  bool acted;
  /* Whether a point was found underneath the mouse. */
  bool found_point;
  /* Whether multiple selected points should be moved. */
  bool multi_point;
  /* Whether a point has already been selected. */
  bool selection_made;
  /* Offset between center of selected points and mouse. */
  bool offset_calc;
  float move_offset[2];
  /* Data about found point. Used for closing splines. */
  Nurb *nu;
  BezTriple *bezt;
  BPoint *bp;
} CurvePenData;

/* Enum to choose between shortcuts for the extra functionalities. */
typedef enum eExtra_key {
  NONE = 0,
  SHIFT = 1,
  CTRL = 2,
  ALT = 3,
  CTRL_SHIFT = 4,
  CTRL_ALT = 5,
  SHIFT_ALT = 6
} eExtra_key;

static const EnumPropertyItem prop_extra_key_types[] = {
    {NONE, "NONE", 0, "None", ""},
    {SHIFT, "SHIFT", 0, "Shift", ""},
    {CTRL, "CTRL", 0, "Ctrl", ""},
    {ALT, "ALT", 0, "Alt", ""},
    {CTRL_SHIFT, "CTRL_SHIFT", 0, "Ctrl-Shift", ""},
    {CTRL_ALT, "CTRL_ALT", 0, "Ctrl-Alt", ""},
    {SHIFT_ALT, "SHIFT_ALT", 0, "Shift-Alt", ""},
    {0, NULL, 0, NULL, NULL},
};

static const EnumPropertyItem prop_handle_types[] = {
    {HD_AUTO, "AUTO", 0, "Auto", ""},
    {HD_VECT, "VECTOR", 0, "Vector", ""},
    {0, NULL, 0, NULL, NULL},
};

typedef enum eClose_opt {
  OFF = 0,
  ON_PRESS = 1,
  ON_CLICK = 2,
} eExtra_key;

static const EnumPropertyItem prop_close_spline_opts[] = {
    {OFF, "OFF", 0, "None", ""},
    {ON_PRESS, "ON_PRESS", 0, "On Press", "Move handles after closing the spline"},
    {ON_CLICK, "ON_CLICK", 0, "On Click", "Spline closes on release if not dragged"},
    {0, NULL, 0, NULL, NULL},
};

static void update_location_for_2d_curve(const ViewContext *vc, float location[3])
{
  Curve *cu = vc->obedit->data;
  if (CU_IS_2D(cu)) {
    const float eps = 1e-6f;

    /* get the view vector to 'location' */
    float view_dir[3];
    ED_view3d_global_to_vector(vc->rv3d, location, view_dir);

    /* get the plane */
    float plane[4];
    /* only normalize to avoid precision errors */
    normalize_v3_v3(plane, vc->obedit->obmat[2]);
    plane[3] = -dot_v3v3(plane, vc->obedit->obmat[3]);

    if (fabsf(dot_v3v3(view_dir, plane)) < eps) {
      /* can't project on an aligned plane. */
    }
    else {
      float lambda;
      if (isect_ray_plane_v3(location, view_dir, plane, &lambda, false)) {
        /* check if we're behind the viewport */
        float location_test[3];
        madd_v3_v3v3fl(location_test, location, view_dir, lambda);
        if ((vc->rv3d->is_persp == false) ||
            (mul_project_m4_v3_zfac(vc->rv3d->persmat, location_test) > 0.0f)) {
          copy_v3_v3(location, location_test);
        }
      }
    }
  }

  float imat[4][4];
  invert_m4_m4(imat, vc->obedit->obmat);
  mul_m4_v3(imat, location);

  if (CU_IS_2D(cu)) {
    location[2] = 0.0f;
  }
}

static void screenspace_to_worldspace(const float pos_2d[2],
                                      const float depth[3],
                                      const ViewContext *vc,
                                      float r_pos_3d[3])
{
  mul_v3_m4v3(r_pos_3d, vc->obedit->obmat, depth);
  ED_view3d_win_to_3d(vc->v3d, vc->region, r_pos_3d, pos_2d, r_pos_3d);
  update_location_for_2d_curve(vc, r_pos_3d);
}

static void screenspace_to_worldspace_int(const int pos_2d[2],
                                          const float depth[3],
                                          const ViewContext *vc,
                                          float r_pos_3d[3])
{
  const float pos_2d_fl[2] = {UNPACK2(pos_2d)};
  screenspace_to_worldspace(pos_2d_fl, depth, vc, r_pos_3d);
}

static bool worldspace_to_screenspace(const float pos_3d[3],
                                      const ViewContext *vc,
                                      float r_pos_2d[2])
{
  return ED_view3d_project_float_object(
             vc->region, pos_3d, r_pos_2d, V3D_PROJ_RET_CLIP_BB | V3D_PROJ_RET_CLIP_WIN) ==
         V3D_PROJ_RET_OK;
}

static void move_bezt_by_change(BezTriple *bezt, const float change[3])
{
  add_v3_v3(bezt->vec[0], change);
  add_v3_v3(bezt->vec[1], change);
  add_v3_v3(bezt->vec[2], change);
}

/* Move entire control point to given worldspace location. */
static void move_bezt_to_location(BezTriple *bezt, const float location[3])
{
  float change[3];
  sub_v3_v3v3(change, location, bezt->vec[1]);
  move_bezt_by_change(bezt, change);
}

/* Alter handle types to allow free movement (Set handles to #FREE or #ALIGN). */
static void remove_handle_movement_constraints(BezTriple *bezt, const bool f1, const bool f3)
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

static void move_bezt_handle_or_vertex_to_location(BezTriple *bezt,
                                                   const float mval[2],
                                                   const short cp_index,
                                                   const ViewContext *vc)
{
  float location[3];
  screenspace_to_worldspace(mval, bezt->vec[cp_index], vc, location);
  if (cp_index == 1) {
    move_bezt_to_location(bezt, location);
  }
  else {
    copy_v3_v3(bezt->vec[cp_index], location);
    if (bezt->h1 == HD_ALIGN && bezt->h2 == HD_ALIGN) {
      float handle_vec[3];
      sub_v3_v3v3(handle_vec, bezt->vec[1], location);
      const short other_handle = cp_index == 2 ? 0 : 2;
      normalize_v3_length(handle_vec, len_v3v3(bezt->vec[1], bezt->vec[other_handle]));
      add_v3_v3v3(bezt->vec[other_handle], bezt->vec[1], handle_vec);
    }
  }
}

static void move_bp_to_location(BPoint *bp, const float mval[2], const ViewContext *vc)
{
  float location[3];
  screenspace_to_worldspace(mval, bp->vec, vc, location);

  copy_v3_v3(bp->vec, location);
}

static bool get_selected_center(const ListBase *nurbs,
                                float r_center[3],
                                const bool mid_only,
                                const bool bezt_only)
{
  int end_count = 0;
  zero_v3(r_center);
  LISTBASE_FOREACH (Nurb *, nu1, nurbs) {
    if (nu1->type == CU_BEZIER) {
      for (int i = 0; i < nu1->pntsu; i++) {
        BezTriple *bezt = nu1->bezt + i;
        if (mid_only) {
          if (BEZT_ISSEL_ANY(bezt)) {
            add_v3_v3(r_center, bezt->vec[1]);
            end_count++;
          }
        }
        else {
          if (BEZT_ISSEL_IDX(bezt, 1)) {
            add_v3_v3(r_center, bezt->vec[1]);
            end_count++;
          }
          else if (BEZT_ISSEL_IDX(bezt, 0)) {
            add_v3_v3(r_center, bezt->vec[0]);
            end_count++;
          }
          else if (BEZT_ISSEL_IDX(bezt, 2)) {
            add_v3_v3(r_center, bezt->vec[2]);
            end_count++;
          }
        }
      }
    }
    else if (!bezt_only) {
      for (int i = 0; i < nu1->pntsu; i++) {
        if ((nu1->bp + i)->f1 & SELECT) {
          add_v3_v3(r_center, (nu1->bp + i)->vec);
          end_count++;
        }
      }
    }
  }
  if (end_count) {
    mul_v3_fl(r_center, 1.0f / end_count);
    return true;
  }
  return false;
}

/* Move all selected points by an amount equivalent to the distance moved by mouse. */
static void move_all_selected_points(ListBase *nurbs,
                                     const bool bezt_only,
                                     const bool move_entire,
                                     CurvePenData *cpd,
                                     const wmEvent *event,
                                     const ViewContext *vc)
{
  float center[3] = {0.0f, 0.0f, 0.0f};
  float change[2], center_2d[2];
  get_selected_center(nurbs, center, false, bezt_only);
  worldspace_to_screenspace(center, vc, center_2d);
  float mval[2] = {UNPACK2(event->xy)};
  sub_v2_v2v2(change, mval, center_2d);
  if (!cpd->offset_calc) {
    float prev_mval[2] = {UNPACK2(event->prev_xy)};
    sub_v2_v2v2(cpd->move_offset, center_2d, prev_mval);
    cpd->offset_calc = true;
  }
  add_v2_v2(change, cpd->move_offset);

  const bool link_handles = cpd->link_handles;
  const bool lock_angle = cpd->lock_angle;

  float change_len = 0.0f;
  if (lock_angle) {
    float mval_3d[3], center_mid[3];
    get_selected_center(nurbs, center_mid, true, true);
    screenspace_to_worldspace_int(event->mval, center_mid, vc, mval_3d);
    change_len = len_v3v3(center_mid, mval_3d);
  }

  LISTBASE_FOREACH (Nurb *, nu, nurbs) {
    if (nu->type == CU_BEZIER) {
      for (int i = 0; i < nu->pntsu; i++) {
        BezTriple *bezt = nu->bezt + i;
        if (BEZT_ISSEL_IDX(bezt, 1) || (move_entire && BEZT_ISSEL_ANY(bezt))) {
          float pos[2], dst[2];
          worldspace_to_screenspace(bezt->vec[1], vc, pos);
          add_v2_v2v2(dst, pos, change);
          move_bezt_handle_or_vertex_to_location(bezt, dst, 1, vc);
        }
        else {
          remove_handle_movement_constraints(
              bezt, BEZT_ISSEL_IDX(bezt, 0), BEZT_ISSEL_IDX(bezt, 2));
          if (BEZT_ISSEL_IDX(bezt, 0)) {
            if (lock_angle) {
              float change_3d[3];
              sub_v3_v3v3(change_3d, bezt->vec[0], bezt->vec[1]);
              normalize_v3_length(change_3d, change_len);
              add_v3_v3v3(bezt->vec[0], bezt->vec[1], change_3d);
            }
            else {
              float pos[2], dst[2];
              worldspace_to_screenspace(bezt->vec[0], vc, pos);
              add_v2_v2v2(dst, pos, change);
              move_bezt_handle_or_vertex_to_location(bezt, dst, 0, vc);
              if (link_handles) {
                float handle[3];
                sub_v3_v3v3(handle, bezt->vec[1], bezt->vec[0]);
                add_v3_v3v3(bezt->vec[2], bezt->vec[1], handle);
              }
            }
          }
          else if (BEZT_ISSEL_IDX(bezt, 2)) {
            if (lock_angle) {
              float change_3d[3];
              sub_v3_v3v3(change_3d, bezt->vec[2], bezt->vec[1]);
              normalize_v3_length(change_3d, change_len);
              add_v3_v3v3(bezt->vec[2], bezt->vec[1], change_3d);
            }
            else {
              float pos[2], dst[2];
              worldspace_to_screenspace(bezt->vec[2], vc, pos);
              add_v2_v2v2(dst, pos, change);
              move_bezt_handle_or_vertex_to_location(bezt, dst, 2, vc);
              if (link_handles) {
                float handle[3];
                sub_v3_v3v3(handle, bezt->vec[1], bezt->vec[2]);
                add_v3_v3v3(bezt->vec[0], bezt->vec[1], handle);
              }
            }
          }
        }
      }
      BKE_nurb_handles_calc(nu);
    }
    else {
      for (int i = 0; i < nu->pntsu; i++) {
        BPoint *bp = nu->bp + i;
        if (bp->f1 & SELECT) {
          float pos[2], dst[2];
          worldspace_to_screenspace(bp->vec, vc, pos);
          add_v2_v2v2(dst, pos, change);
          move_bp_to_location(bp, dst, vc);
        }
      }
    }
  }
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
                                      float *r_fraction)
{
  float pos1_2d[2], pos2_2d[2], vec1[2], vec2[2], vec3[2];

  /* Get screen space coordinates of points. */
  if (!(worldspace_to_screenspace(pos1, vc, pos1_2d) &&
        worldspace_to_screenspace(pos2, vc, pos2_2d))) {
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
    *r_fraction = 1 - dot2 / len_vec3_sq;

    float pos_dif[3];
    sub_v3_v3v3(pos_dif, pos2, pos1);
    madd_v3_v3v3fl(r_point, pos1, pos_dif, *r_fraction);
    return true;
  }

  if (len_manhattan_v2(vec1) < len_manhattan_v2(vec2)) {
    copy_v3_v3(r_point, pos1);
    *r_fraction = 0.0f;
    return false;
  }
  copy_v3_v3(r_point, pos2);
  *r_fraction = 1.0f;
  return false;
}

/* Get closest vertex in all nurbs in given #ListBase to a given point.
 * Returns true if point is found. */
static bool get_closest_vertex_to_point_in_nurbs(const ListBase *nurbs,
                                                 Nurb **r_nu,
                                                 BezTriple **r_bezt,
                                                 BPoint **r_bp,
                                                 short *r_bezt_idx,
                                                 const float point[2],
                                                 const float sel_dist_mul,
                                                 const ViewContext *vc)
{
  *r_nu = NULL;
  *r_bezt = NULL;
  *r_bp = NULL;

  float min_distance_bezt = FLT_MAX;
  float min_distance_bp = FLT_MAX;

  BezTriple *closest_bezt = NULL;
  short closest_handle = 0;
  BPoint *closest_bp = NULL;
  Nurb *closest_bezt_nu = NULL;
  Nurb *closest_bp_nu = NULL;

  LISTBASE_FOREACH (Nurb *, nu, nurbs) {
    if (nu->type == CU_BEZIER) {
      for (int i = 0; i < nu->pntsu; i++) {
        BezTriple *bezt = &nu->bezt[i];
        float bezt_vec[2];
        int start = 0, end = 3;
        int handle_display = vc->v3d->overlay.handle_display;
        if (handle_display == CURVE_HANDLE_NONE ||
            (handle_display == CURVE_HANDLE_SELECTED && !BEZT_ISSEL_ANY(bezt))) {
          start = 1, end = 2;
        }
        for (short j = start; j < end; j++) {
          if (worldspace_to_screenspace(bezt->vec[j], vc, bezt_vec)) {
            const float distance = len_manhattan_v2v2(bezt_vec, point);
            if (distance < min_distance_bezt) {
              min_distance_bezt = distance;
              closest_bezt = bezt;
              closest_bezt_nu = nu;
              closest_handle = j;
            }
          }
        }
      }
    }
    else {
      for (int i = 0; i < nu->pntsu; i++) {
        BPoint *bp = &nu->bp[i];
        float bp_vec[2];
        if (worldspace_to_screenspace(bp->vec, vc, bp_vec)) {
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

  const float threshold_distance = ED_view3d_select_dist_px() * sel_dist_mul;
  if (min_distance_bezt < threshold_distance || min_distance_bp < threshold_distance) {
    if (min_distance_bp < min_distance_bezt) {
      *r_bp = closest_bp;
      *r_nu = closest_bp_nu;
    }
    else {
      *r_bezt = closest_bezt;
      *r_bezt_idx = closest_handle;
      *r_nu = closest_bezt_nu;
    }
    return true;
  }
  return false;
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
static void update_data_if_closest_point_in_segment(const BezTriple *bezt1,
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
    bool check = worldspace_to_screenspace(points + 3 * k, vc, screen_co);

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
    if (worldspace_to_screenspace(point, vc, point_2d)) {
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
        worldspace_to_screenspace(nu->bezt->vec[1], vc, screen_co);
        assign_cut_data(data, FLT_MAX, nu, 0, -1, 0.0f, nu->bezt->vec[1]);
      }

      BezTriple *bezt = NULL;
      for (int i = 0; i < nu->pntsu - 1; i++) {
        bezt = &nu->bezt[i];
        update_data_if_closest_point_in_segment(bezt, bezt + 1, nu, i, vc, data);
      }

      if (nu->flagu & CU_NURB_CYCLIC && bezt) {
        update_data_if_closest_point_in_segment(bezt + 1, nu->bezt, nu, nu->pntsu - 1, vc, data);
      }

      float fraction = 0.0f;
      float point[3];
      if (data->has_next) {
        get_closest_point_on_edge(point, data->mval, data->cut_loc, data->next_loc, vc, &fraction);
        worldspace_to_screenspace(point, vc, screen_co);
        const float dist1 = len_manhattan_v2v2(screen_co, data->mval);
        data->min_dist = dist1;
      }

      if (data->has_prev) {
        get_closest_point_on_edge(point, data->mval, data->cut_loc, data->prev_loc, vc, &fraction);
        worldspace_to_screenspace(point, vc, screen_co);
        const float dist2 = len_manhattan_v2v2(screen_co, data->mval);
        if (dist2 < data->min_dist) {
          data->min_dist = dist2;
          fraction = -fraction;
        }
      }

      data->parameter += fraction / nu->resolu;
    }
    else {
      float screen_co[2];
      if (data->nurb == NULL) {
        worldspace_to_screenspace(nu->bp->vec, vc, screen_co);
        assign_cut_data(
            data, len_manhattan_v2v2(screen_co, data->mval), nu, -1, 0, 0.0f, nu->bp->vec);
      }

      int end = nu->pntsu - 1;
      if (nu->flagu & CU_NURB_CYCLIC) {
        end = nu->pntsu;
      }

      for (int i = 0; i < end; i++) {
        float point[3], factor;
        int next_i = (i + 1) % nu->pntsu;
        bool found_min = get_closest_point_on_edge(
            point, data->mval, (nu->bp + i)->vec, (nu->bp + next_i)->vec, vc, &factor);
        if (found_min) {
          float point_2d[2];
          if (worldspace_to_screenspace(point, vc, point_2d)) {
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
static void insert_bezt_to_nurb(Nurb *nu, const CutData *data, Curve *cu)
{
  EditNurb *editnurb = cu->editnurb;

  BezTriple *bezt1 = (BezTriple *)MEM_mallocN((nu->pntsu + 1) * sizeof(BezTriple), __func__);
  const int index = data->bezt_index + 1;
  /* Copy all control points before the cut to the new memory. */
  ED_curve_beztcpy(editnurb, bezt1, nu->bezt, index);
  BezTriple *new_bezt = bezt1 + index;

  /* Duplicate control point after the cut. */
  ED_curve_beztcpy(editnurb, new_bezt, new_bezt - 1, 1);
  copy_v3_v3(new_bezt->vec[1], data->cut_loc);

  if (index < nu->pntsu) {
    /* Copy all control points after the cut to the new memory. */
    ED_curve_beztcpy(editnurb, bezt1 + index + 1, nu->bezt + index, nu->pntsu - index);
  }

  nu->pntsu += 1;
  cu->actvert = index;
  cu->actnu = get_nurb_index(BKE_curve_editNurbs_get(cu), nu);

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
static void insert_bp_to_nurb(Nurb *nu, const CutData *data, Curve *cu)
{
  EditNurb *editnurb = cu->editnurb;

  BPoint *bp1 = (BPoint *)MEM_mallocN((nu->pntsu + 1) * sizeof(BPoint), __func__);
  const int index = data->bp_index + 1;
  /* Copy all control points before the cut to the new memory. */
  ED_curve_bpcpy(editnurb, bp1, nu->bp, index);
  BPoint *new_bp = bp1 + index;

  /* Duplicate control point after the cut. */
  ED_curve_bpcpy(editnurb, new_bp, new_bp - 1, 1);
  copy_v3_v3(new_bp->vec, data->cut_loc);

  if (index < nu->pntsu) {
    /* Copy all control points after the cut to the new memory. */
    ED_curve_bpcpy(editnurb, bp1 + index + 1, nu->bp + index, (nu->pntsu - index));
  }

  nu->pntsu += 1;
  cu->actvert = index;
  cu->actnu = get_nurb_index(BKE_curve_editNurbs_get(cu), nu);

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
  new_bp->f1 |= SELECT;
}

/* Make a cut on the nearest nurb at the closest point. Return true if spline is nearby. */
static bool insert_point_to_segment(
    const wmEvent *event, Curve *cu, Nurb **r_nu, const float sel_dist_mul, const ViewContext *vc)
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

  const float threshold_distance = ED_view3d_select_dist_px() * sel_dist_mul;
  Nurb *nu = data.nurb;
  if (nu) {
    if (nu->type == CU_BEZIER) {
      update_cut_loc_in_data(&data, vc);
      if (data.min_dist < threshold_distance) {
        insert_bezt_to_nurb(nu, &data, cu);
        *r_nu = nu;
        return true;
      }
    }
    else if (data.min_dist < threshold_distance) {
      insert_bp_to_nurb(nu, &data, cu);
      *r_nu = nu;
      return true;
    }
  }
  return false;
}

static void get_selected_points(
    Curve *cu, View3D *v3d, Nurb **r_nu, BezTriple **r_bezt, BPoint **r_bp)
{
  /* In nu and (bezt or bp) selected are written if there's 1 sel. */
  /* If more points selected in 1 spline: return only nu, bezt and bp are 0. */
  ListBase *nurbs = &cu->editnurb->nurbs;
  BezTriple *bezt1;
  BPoint *bp1;
  int a;

  *r_nu = NULL;
  *r_bezt = NULL;
  *r_bp = NULL;

  LISTBASE_FOREACH (Nurb *, nu1, nurbs) {
    if (nu1->type == CU_BEZIER) {
      bezt1 = nu1->bezt;
      a = nu1->pntsu;
      while (a--) {
        if (BEZT_ISSEL_ANY_HIDDENHANDLES(v3d, bezt1)) {
          if (*r_bezt || *r_bp) {
            *r_bp = NULL;
            *r_bezt = NULL;
            return;
          }
          *r_bezt = bezt1;
          *r_nu = nu1;
        }
        bezt1++;
      }
    }
    else {
      bp1 = nu1->bp;
      a = nu1->pntsu * nu1->pntsv;
      while (a--) {
        if (bp1->f1 & SELECT) {
          if (*r_bezt || *r_bp) {
            *r_bp = NULL;
            *r_bezt = NULL;
            return;
          }
          *r_bp = bp1;
          *r_nu = nu1;
        }
        bp1++;
      }
    }
  }
}

static void extrude_vertices_from_selected_endpoints(EditNurb *editnurb,
                                                     ListBase *nurbs,
                                                     Curve *cu,
                                                     const float change[3])
{
  int nu_index = 0;
  LISTBASE_FOREACH (Nurb *, nu1, nurbs) {
    if (nu1->type == CU_BEZIER) {
      BezTriple *last_bezt = nu1->bezt + nu1->pntsu - 1;
      const bool first_sel = BEZT_ISSEL_ANY(nu1->bezt);
      const bool last_sel = BEZT_ISSEL_ANY(last_bezt) && nu1->pntsu > 1;
      if (first_sel) {
        if (last_sel) {
          BezTriple *new_bezt = (BezTriple *)MEM_mallocN((nu1->pntsu + 2) * sizeof(BezTriple),
                                                         __func__);
          ED_curve_beztcpy(editnurb, new_bezt, nu1->bezt, 1);
          ED_curve_beztcpy(editnurb, new_bezt + nu1->pntsu + 1, last_bezt, 1);
          BEZT_DESEL_ALL(nu1->bezt);
          BEZT_DESEL_ALL(last_bezt);
          ED_curve_beztcpy(editnurb, new_bezt + 1, nu1->bezt, nu1->pntsu);

          move_bezt_by_change(new_bezt, change);
          move_bezt_by_change(new_bezt + nu1->pntsu + 1, change);
          MEM_freeN(nu1->bezt);
          nu1->bezt = new_bezt;
          nu1->pntsu += 2;
        }
        else {
          BezTriple *new_bezt = (BezTriple *)MEM_mallocN((nu1->pntsu + 1) * sizeof(BezTriple),
                                                         __func__);
          ED_curve_beztcpy(editnurb, new_bezt, nu1->bezt, 1);
          BEZT_DESEL_ALL(nu1->bezt);
          ED_curve_beztcpy(editnurb, new_bezt + 1, nu1->bezt, nu1->pntsu);
          move_bezt_by_change(new_bezt, change);
          MEM_freeN(nu1->bezt);
          nu1->bezt = new_bezt;
          nu1->pntsu++;
        }
        cu->actnu = nu_index;
      }
      else if (last_sel) {
        BezTriple *new_bezt = (BezTriple *)MEM_mallocN((nu1->pntsu + 1) * sizeof(BezTriple),
                                                       __func__);
        ED_curve_beztcpy(editnurb, new_bezt + nu1->pntsu, last_bezt, 1);
        BEZT_DESEL_ALL(last_bezt);
        ED_curve_beztcpy(editnurb, new_bezt, nu1->bezt, nu1->pntsu);
        move_bezt_by_change(new_bezt + nu1->pntsu, change);
        MEM_freeN(nu1->bezt);
        nu1->bezt = new_bezt;
        nu1->pntsu++;
        cu->actnu = nu_index;
      }
    }
    else {
      BPoint *last_bp = nu1->bp + nu1->pntsu - 1;
      const bool first_sel = nu1->bp->f1 & SELECT;
      const bool last_sel = last_bp->f1 & SELECT;
      if (first_sel) {
        if (last_sel) {
          BPoint *new_bp = (BPoint *)MEM_mallocN((nu1->pntsu + 2) * sizeof(BPoint), __func__);
          ED_curve_bpcpy(editnurb, new_bp, nu1->bp, 1);
          ED_curve_bpcpy(editnurb, new_bp + nu1->pntsu + 1, last_bp, 1);
          new_bp->f1 &= ~SELECT;
          last_bp->f1 &= ~SELECT;
          ED_curve_bpcpy(editnurb, new_bp + 1, nu1->bp, nu1->pntsu);
          add_v3_v3(new_bp->vec, change);
          add_v3_v3((new_bp + nu1->pntsu + 1)->vec, change);
          MEM_freeN(nu1->bp);
          nu1->bp = new_bp;
          nu1->pntsu += 2;
        }
        else {
          BPoint *new_bp = (BPoint *)MEM_mallocN((nu1->pntsu + 1) * sizeof(BPoint), __func__);
          ED_curve_bpcpy(editnurb, new_bp, nu1->bp, 1);
          new_bp->f1 &= ~SELECT;
          ED_curve_bpcpy(editnurb, new_bp + 1, nu1->bp, nu1->pntsu);
          add_v3_v3(new_bp->vec, change);
          MEM_freeN(nu1->bp);
          nu1->bp = new_bp;
          nu1->pntsu++;
        }
        BKE_nurb_knot_calc_u(nu1);
        cu->actnu = nu_index;
      }
      else if (last_sel) {
        BPoint *new_bp = (BPoint *)MEM_mallocN((nu1->pntsu + 1) * sizeof(BPoint), __func__);
        ED_curve_bpcpy(editnurb, new_bp, nu1->bp, nu1->pntsu);
        ED_curve_bpcpy(editnurb, new_bp + nu1->pntsu, last_bp, 1);
        last_bp->f1 &= ~SELECT;
        ED_curve_bpcpy(editnurb, new_bp, nu1->bp, nu1->pntsu);
        add_v3_v3((new_bp + nu1->pntsu)->vec, change);
        MEM_freeN(nu1->bp);
        nu1->bp = new_bp;
        nu1->pntsu++;
        BKE_nurb_knot_calc_u(nu1);
        cu->actnu = nu_index;
      }
    }
    nu_index++;
  }
}

static void deselect_all_center_vertices(ListBase *nurbs)
{
  LISTBASE_FOREACH (Nurb *, nu1, nurbs) {
    if (nu1->pntsu > 1) {
      int start, end;
      if (nu1->flagu & CU_NURB_CYCLIC) {
        start = 0;
        end = nu1->pntsu;
      }
      else {
        start = 1;
        end = nu1->pntsu - 1;
      }
      for (int i = start; i < end; i++) {
        if (nu1->type == CU_BEZIER) {
          BEZT_DESEL_ALL(nu1->bezt + i);
        }
        else {
          (nu1->bp + i)->f1 &= ~SELECT;
        }
      }
    }
  }
}

static bool is_last_bezt(const Nurb *nu, const BezTriple *bezt)
{
  return nu->pntsu > 1 && nu->bezt + nu->pntsu - 1 == bezt && !(nu->flagu & CU_NURB_CYCLIC);
}

/* Add new vertices connected to the selected vertices. */
static void extrude_points_from_selected_vertices(const ViewContext *vc,
                                                  Object *obedit,
                                                  const wmEvent *event,
                                                  const bool extrude_center,
                                                  const int extrude_handle)
{
  Curve *cu = vc->obedit->data;
  ListBase *nurbs = BKE_curve_editNurbs_get(cu);
  float center[3] = {0.0f, 0.0f, 0.0f};
  if (!extrude_center) {
    deselect_all_center_vertices(nurbs);
  }
  bool sel_exists = get_selected_center(nurbs, center, true, false);

  float location[3];
  if (sel_exists) {
    mul_v3_m4v3(location, vc->obedit->obmat, center);
  }
  else {
    copy_v3_v3(location, vc->scene->cursor.location);
  }

  ED_view3d_win_to_3d_int(vc->v3d, vc->region, location, event->mval, location);

  update_location_for_2d_curve(vc, location);
  EditNurb *editnurb = cu->editnurb;

  if (!extrude_center && sel_exists) {
    float change[3];
    sub_v3_v3v3(change, location, center);
    /* Reimplemenented due to unexpected behavior for extrusion of 2-point spline. */
    extrude_vertices_from_selected_endpoints(editnurb, nurbs, cu, change);
  }
  else {
    Nurb *old_last_nu = editnurb->nurbs.last;
    ed_editcurve_addvert(cu, editnurb, vc->v3d, location);
    Nurb *new_last_nu = editnurb->nurbs.last;

    if (old_last_nu != new_last_nu) {
      new_last_nu->flagu = ~CU_NURB_CYCLIC;
    }
  }

  FOREACH_SELECTED_BEZT_BEGIN(bezt, &cu->editnurb->nurbs)
  if (bezt) {
    bezt->h1 = extrude_handle;
    bezt->h2 = extrude_handle;
  }
  FOREACH_SELECTED_BEZT_END
}

/* Check if a spline segment is nearby. */
static bool is_spline_nearby(ViewContext *vc, struct wmOperator *op, const wmEvent *event)
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
  const float sel_dist_mul = RNA_float_get(op->ptr, "sel_dist_mul");
  const float threshold_distance = ED_view3d_select_dist_px() * sel_dist_mul;

  if (data.min_dist < threshold_distance) {
    if (data.nurb && (data.nurb->type == CU_BEZIER) && RNA_boolean_get(op->ptr, "move_segment")) {
      MoveSegmentData *seg_data;
      CurvePenData *cpd = (CurvePenData *)(op->customdata);
      cpd->msd = seg_data = MEM_callocN(sizeof(MoveSegmentData), __func__);
      seg_data->bezt_index = data.bezt_index;
      seg_data->nu = data.nurb;
      seg_data->t = data.parameter;
    }
    return true;
  }
  return false;
}

static bool is_extra_key_pressed(const wmEvent *event, int key)
{
  return (key == SHIFT && event->shift && !event->ctrl && !event->alt) ||
         (key == CTRL && !event->shift && event->ctrl && !event->alt) ||
         (key == ALT && !event->shift && !event->ctrl && event->alt) ||
         (key == CTRL_SHIFT && event->shift && event->ctrl && !event->alt) ||
         (key == CTRL_ALT && !event->shift && event->ctrl && event->alt) ||
         (key == SHIFT_ALT && event->shift && !event->ctrl && event->alt);
}

/* Move segment to mouse pointer. */
static void move_segment(MoveSegmentData *seg_data, const wmEvent *event, ViewContext *vc)
{
  Nurb *nu = seg_data->nu;
  BezTriple *bezt1 = nu->bezt + seg_data->bezt_index;
  BezTriple *bezt2 = BKE_nurb_bezt_get_next(nu, bezt1);

  const float t = max_ff(min_ff(seg_data->t, 0.9f), 0.1f);

  const float t_sq = t * t;
  const float t_cu = t_sq * t;
  const float one_minus_t = 1 - t;
  const float one_minus_t_sq = one_minus_t * one_minus_t;
  const float one_minus_t_cu = one_minus_t_sq * one_minus_t;

  float mouse_3d[3];
  float depth[3];
  /* Use the center of the spline segment as depth. */
  get_bezier_interpolated_point(depth, bezt1, bezt2, t);
  screenspace_to_worldspace_int(event->mval, depth, vc, mouse_3d);

  /*
   * Equation of Bezier Curve
   *      => B(t) = (1-t)^3 * P0 + 3(1-t)^2 * t * P1 + 3(1-t) * t^2 * P2 + t^3 * P3
   *
   * Mouse location (Say Pm) should satisfy this equation.
   * Therefore => (1/t - 1) * P1 + P2 = (Pm - (1 - t)^3 * P0 - t^3 * P3) / [3 * (1 - t) * t^2] = k1
   * (in code)
   *
   * Another constraint is required to identify P1 and P2.
   * The constraint used is that the vector between P1 and P2 doesn't change.
   * Therefore => P1 - P2 = k2
   *
   * From the two equations => P1 = t(k1 + k2) and P2 = P1 - K2
   */

  float k1[3];
  const float denom = (3.0f * one_minus_t * t_sq);
  k1[0] = (mouse_3d[0] - one_minus_t_cu * bezt1->vec[1][0] - t_cu * bezt2->vec[1][0]) / denom;
  k1[1] = (mouse_3d[1] - one_minus_t_cu * bezt1->vec[1][1] - t_cu * bezt2->vec[1][1]) / denom;
  k1[2] = (mouse_3d[2] - one_minus_t_cu * bezt1->vec[1][2] - t_cu * bezt2->vec[1][2]) / denom;

  float k2[3];
  sub_v3_v3v3(k2, bezt1->vec[2], bezt2->vec[0]);

  /* P1 = t(k1 + k2) */
  add_v3_v3v3(bezt1->vec[2], k1, k2);
  mul_v3_fl(bezt1->vec[2], t);
  /* P2 = P1 - K2 */
  sub_v3_v3v3(bezt2->vec[0], bezt1->vec[2], k2);

  remove_handle_movement_constraints(bezt1, true, true);
  remove_handle_movement_constraints(bezt2, true, true);

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

/* Toggle between `free` and `align` handles of the given `BezTriple` */
static void toggle_bezt_free_align_handles(ListBase *nurbs)
{
  FOREACH_SELECTED_BEZT_BEGIN(bezt, nurbs)
  if (bezt->h1 != HD_FREE || bezt->h2 != HD_FREE) {
    bezt->h1 = bezt->h2 = HD_FREE;
  }
  else {
    bezt->h1 = bezt->h2 = HD_ALIGN;
  }

  BKE_nurb_handles_calc(nu);
  FOREACH_SELECTED_BEZT_END
}

/* Returns true if point was found under mouse. */
static bool delete_point_under_mouse(ViewContext *vc,
                                     const wmEvent *event,
                                     const float sel_dist_mul)
{
  BezTriple *bezt = NULL;
  BPoint *bp = NULL;
  Nurb *nu = NULL;
  short temp = 0;
  Curve *cu = vc->obedit->data;
  ListBase *nurbs = BKE_curve_editNurbs_get(cu);
  const float mouse_point[2] = {UNPACK2(event->mval)};

  get_closest_vertex_to_point_in_nurbs(
      nurbs, &nu, &bezt, &bp, &temp, mouse_point, sel_dist_mul, vc);
  const bool found_point = nu != NULL;

  bool deleted = false;
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
      deleted = true;
      cu->actvert = CU_ACT_NONE;
    }
  }

  if (nu && nu->type == CU_BEZIER) {
    BKE_nurb_handles_calc(nu);
  }

  return deleted;
}

static void move_adjacent_handle(ViewContext *vc, const wmEvent *event, ListBase *nurbs)
{
  FOREACH_SELECTED_BEZT_BEGIN(bezt, nurbs)
  BezTriple *adj_bezt;
  int cp_idx;
  if (nu->pntsu == 1) {
    continue;
  }
  if (nu->bezt == bezt) {
    adj_bezt = BKE_nurb_bezt_get_next(nu, bezt);
    cp_idx = 0;
  }
  else if (nu->bezt + nu->pntsu - 1 == bezt) {
    adj_bezt = BKE_nurb_bezt_get_prev(nu, bezt);
    cp_idx = 2;
  }
  else {
    if (BEZT_ISSEL_IDX(bezt, 0)) {
      adj_bezt = BKE_nurb_bezt_get_prev(nu, bezt);
      cp_idx = 2;
    }
    else if (BEZT_ISSEL_IDX(bezt, 2)) {
      adj_bezt = BKE_nurb_bezt_get_next(nu, bezt);
      cp_idx = 0;
    }
    else {
      continue;
    }
  }
  adj_bezt->h1 = adj_bezt->h2 = HD_FREE;

  float screen_co[2];
  int displacement[2];
  /* Get the screen space coordinates of moved handle. */
  ED_view3d_project_float_object(
      vc->region, adj_bezt->vec[cp_idx], screen_co, V3D_PROJ_RET_CLIP_BB | V3D_PROJ_RET_CLIP_WIN);
  sub_v2_v2v2_int(displacement, event->xy, event->prev_xy);
  const float disp_fl[2] = {UNPACK2(displacement)};

  /* Add the displacement of the mouse to the handle position. */
  add_v2_v2v2(screen_co, screen_co, disp_fl);
  move_bezt_handle_or_vertex_to_location(adj_bezt, screen_co, cp_idx, vc);
  BKE_nurb_handles_calc(nu);
  FOREACH_SELECTED_BEZT_END
}

/* Close the spline if endpoints are selected consecutively. Return true if cycle was created. */
static bool make_cyclic_if_endpoints(Nurb *sel_nu,
                                     BezTriple *sel_bezt,
                                     BPoint *sel_bp,
                                     ViewContext *vc,
                                     bContext *C,
                                     const float sel_dist_mul)
{
  if (sel_bezt || (sel_bp && sel_nu->pntsu > 2)) {
    const bool is_bezt_endpoint = (sel_nu->type == CU_BEZIER &&
                                   (sel_bezt == sel_nu->bezt ||
                                    sel_bezt == sel_nu->bezt + sel_nu->pntsu - 1));
    const bool is_bp_endpoint = (sel_nu->type != CU_BEZIER &&
                                 (sel_bp == sel_nu->bp ||
                                  sel_bp == sel_nu->bp + sel_nu->pntsu - 1));
    if (!(is_bezt_endpoint || is_bp_endpoint)) {
      return false;
    }

    Nurb *nu = NULL;
    BezTriple *bezt = NULL;
    BPoint *bp = NULL;
    Curve *cu = vc->obedit->data;
    short bezt_idx;
    const float mval_fl[2] = {UNPACK2(vc->mval)};

    get_closest_vertex_to_point_in_nurbs(
        &(cu->editnurb->nurbs), &nu, &bezt, &bp, &bezt_idx, mval_fl, sel_dist_mul, vc);

    if (nu == sel_nu &&
        ((nu->type == CU_BEZIER && bezt != sel_bezt &&
          (bezt == nu->bezt || bezt == nu->bezt + nu->pntsu - 1) && bezt_idx == 1) ||
         (nu->type != CU_BEZIER && bp != sel_bp &&
          (bp == nu->bp || bp == nu->bp + nu->pntsu - 1)))) {
      View3D *v3d = CTX_wm_view3d(C);
      ListBase *nurbs = object_editcurve_get(vc->obedit);
      ed_curve_toggle_cyclic(v3d, nurbs, 0);
      return true;
    }
  }
  return false;
}

static void toggle_select_bezt(BezTriple *bezt, const short bezt_idx)
{
  if (bezt_idx == 1) {
    if (BEZT_ISSEL_IDX(bezt, 1)) {
      BEZT_DESEL_ALL(bezt);
    }
    else {
      BEZT_SEL_ALL(bezt);
    }
  }
  else {
    if (BEZT_ISSEL_IDX(bezt, bezt_idx)) {
      BEZT_DESEL_IDX(bezt, bezt_idx);
    }
    else {
      BEZT_SEL_IDX(bezt, bezt_idx);
    }
  }
}

static void toggle_select_bp(BPoint *bp)
{
  if (bp->f1 & SELECT) {
    bp->f1 &= ~SELECT;
  }
  else {
    bp->f1 |= SELECT;
  }
}

static void toggle_handle_types(BezTriple *bezt, short bezt_idx, CurvePenData *cpd)
{
  if (bezt_idx == 0) {
    if (bezt->h1 == HD_VECT) {
      bezt->h1 = bezt->h2 = HD_AUTO;
    }
    else {
      bezt->h1 = HD_VECT;
      if (bezt->h2 != HD_VECT) {
        bezt->h2 = HD_FREE;
      }
    }
    cpd->acted = true;
  }
  else if (bezt_idx == 2) {
    if (bezt->h2 == HD_VECT) {
      bezt->h1 = bezt->h2 = HD_AUTO;
    }
    else {
      bezt->h2 = HD_VECT;
      if (bezt->h1 != HD_VECT) {
        bezt->h1 = HD_FREE;
      }
    }
    cpd->acted = true;
  }
}

static void cycle_handles(BezTriple *bezt)
{
  if (bezt->h1 == HD_AUTO) {
    bezt->h1 = bezt->h2 = HD_VECT;
  }
  else if (bezt->h1 == HD_VECT) {
    bezt->h1 = bezt->h2 = HD_ALIGN;
  }
  else if (bezt->h1 == HD_ALIGN) {
    bezt->h1 = bezt->h2 = HD_FREE;
  }
  else {
    bezt->h1 = bezt->h2 = HD_AUTO;
  }
}

static int curve_pen_modal(bContext *C, wmOperator *op, const wmEvent *event)
{
  Depsgraph *depsgraph = CTX_data_ensure_evaluated_depsgraph(C);
  ViewContext vc;
  Object *obedit = CTX_data_edit_object(C);

  ED_view3d_viewcontext_init(C, &vc, depsgraph);
  Curve *cu = vc.obedit->data;
  ListBase *nurbs = &cu->editnurb->nurbs;

  BezTriple *bezt = NULL;
  BPoint *bp = NULL;
  Nurb *nu = NULL;

  int ret = OPERATOR_RUNNING_MODAL;
  /* Distance threshold for mouse clicks to affect the spline or its points */
  const float sel_dist_mul = RNA_float_get(op->ptr, "sel_dist_mul");

  CurvePenData *cpd;
  if (op->customdata == NULL) {
    op->customdata = cpd = MEM_callocN(sizeof(CurvePenData), __func__);
  }
  else {
    cpd = (CurvePenData *)(op->customdata);
  }
  const float mval_fl[2] = {UNPACK2(event->mval)};

  const bool extrude_point = RNA_boolean_get(op->ptr, "extrude_point");
  const bool extrude_center = RNA_boolean_get(op->ptr, "extrude_center");
  const bool new_spline = RNA_boolean_get(op->ptr, "new_spline");
  const bool delete_point = RNA_boolean_get(op->ptr, "delete_point");
  const bool insert_point = RNA_boolean_get(op->ptr, "insert_point");
  const bool move_seg = RNA_boolean_get(op->ptr, "move_segment");
  const bool select_point = RNA_boolean_get(op->ptr, "select_point");
  const bool select_multi = RNA_boolean_get(op->ptr, "select_multi");
  const bool move_point = RNA_boolean_get(op->ptr, "move_point");
  const bool close_spline = RNA_boolean_get(op->ptr, "close_spline");
  const bool toggle_vector = RNA_boolean_get(op->ptr, "toggle_vector");
  const bool cycle_handle_type = RNA_boolean_get(op->ptr, "cycle_handle_type");
  const int close_spline_opts = RNA_enum_get(op->ptr, "close_spline_opts");
  const int extrude_handle = RNA_enum_get(op->ptr, "extrude_handle");
  const int free_toggle = RNA_enum_get(op->ptr, "free_toggle");
  const int adj_handle = RNA_enum_get(op->ptr, "adj_handle");
  const int move_entire = RNA_enum_get(op->ptr, "move_entire");
  const int link_handles = RNA_enum_get(op->ptr, "link_handles");
  const int lock_angle = RNA_enum_get(op->ptr, "lock_angle");

  if (!cpd->free_toggle_pressed && is_extra_key_pressed(event, free_toggle)) {
    toggle_bezt_free_align_handles(nurbs);
    cpd->dragging = true;
    cpd->link_handles = false;
  }
  cpd->free_toggle_pressed = is_extra_key_pressed(event, free_toggle);

  if (!cpd->link_handles_pressed && is_extra_key_pressed(event, link_handles)) {
    cpd->link_handles = !cpd->link_handles;
    if (cpd->link_handles) {
      move_all_selected_points(nurbs, false, false, cpd, event, &vc);
    }
    else {
      // Recalculate offset after link handles is turned off
      cpd->offset_calc = false;
    }
  }
  cpd->link_handles_pressed = is_extra_key_pressed(event, link_handles);
  cpd->lock_angle = is_extra_key_pressed(event, lock_angle);

  const bool move_entire_pressed = is_extra_key_pressed(event, move_entire);

  if (ELEM(event->type, MOUSEMOVE, INBETWEEN_MOUSEMOVE)) {
    if (!cpd->dragging && WM_event_drag_test(event, event->prev_click_xy) &&
        event->val == KM_PRESS) {
      cpd->dragging = true;
      if (cpd->new_point) {
        FOREACH_SELECTED_BEZT_BEGIN(bezt, nurbs)
        bezt->h1 = bezt->h2 = HD_ALIGN;
        copy_v3_v3(bezt->vec[0], bezt->vec[1]);
        copy_v3_v3(bezt->vec[2], bezt->vec[1]);
        BEZT_DESEL_ALL(bezt);
        BEZT_SEL_IDX(bezt, is_last_bezt(nu, bezt) ? 2 : 0);
        FOREACH_SELECTED_BEZT_END
      }
    }
    if (cpd->dragging) {
      if (cpd->spline_nearby && move_seg && cpd->msd != NULL) {
        MoveSegmentData *seg_data = cpd->msd;
        nu = seg_data->nu;
        move_segment(seg_data, event, &vc);
        cpd->acted = true;
      }
      else if (is_extra_key_pressed(event, adj_handle)) {
        move_adjacent_handle(&vc, event, nurbs);
        cpd->acted = true;
      }
      /* If dragging a new control point, move handle point with mouse cursor. Else move entire
       * control point. */
      else if (cpd->new_point) {
        move_all_selected_points(nurbs, true, move_entire_pressed, cpd, event, &vc);
        cpd->acted = true;
      }
      else if ((select_point || move_point) && !cpd->spline_nearby) {
        if (cpd->found_point) {
          if (move_point) {
            move_all_selected_points(nurbs, false, move_entire_pressed, cpd, event, &vc);
            cpd->acted = true;
          }
        }
      }
      if (nu && nu->type == CU_BEZIER) {
        BKE_nurb_handles_calc(nu);
      }
    }
  }
  else if (ELEM(event->type, LEFTMOUSE)) {
    if (ELEM(event->val, KM_PRESS, KM_DBL_CLICK)) {
      get_selected_points(cu, vc.v3d, &nu, &bezt, &bp);
      cpd->nu = nu;
      cpd->bezt = bezt;
      cpd->bp = bp;

      Nurb *nu1;
      BezTriple *bezt1;
      BPoint *bp1;
      short bezt_idx = 0;
      cpd->found_point = get_closest_vertex_to_point_in_nurbs(
          nurbs, &nu1, &bezt1, &bp1, &bezt_idx, mval_fl, sel_dist_mul, &vc);

      if (move_point && nu1 &&
          (bezt || (bezt1 && !BEZT_ISSEL_IDX(bezt1, bezt_idx)) || (bp1 && !(bp1->f1 & SELECT)))) {
        ED_curve_deselect_all(cu->editnurb);
        if (bezt1) {
          if (bezt_idx == 1) {
            BEZT_SEL_ALL(bezt1);
          }
          else {
            BEZT_SEL_IDX(bezt1, bezt_idx);
          }
          cu->actnu = get_nurb_index(nurbs, nu1);
        }
        else if (bp1) {
          bp1->f1 |= SELECT;
          cu->actnu = get_nurb_index(nurbs, nu1);
        }

        cpd->selection_made = true;
      }
      if (cpd->found_point) {
        if (close_spline && close_spline_opts == ON_PRESS && cpd->nu &&
            !(cpd->nu->flagu & CU_NURB_CYCLIC)) {
          copy_v2_v2_int(vc.mval, event->mval);
          cpd->new_point = cpd->acted = cpd->link_handles = make_cyclic_if_endpoints(
              cpd->nu, cpd->bezt, cpd->bp, &vc, C, sel_dist_mul);
        }
      }
      else if (!cpd->acted) {
        if (is_spline_nearby(&vc, op, event)) {
          cpd->spline_nearby = true;

          /* If move segment is disabled, then insert point on key press and set
          "new_point" to true so that the new point's handles can be controlled. */
          if (insert_point && !move_seg) {
            insert_point_to_segment(event, vc.obedit->data, &nu, sel_dist_mul, &vc);
            cpd->new_point = cpd->acted = cpd->link_handles = true;
          }
        }
        else if (new_spline) {
          ED_curve_deselect_all(((Curve *)(vc.obedit->data))->editnurb);
          extrude_points_from_selected_vertices(
              &vc, obedit, event, extrude_center, extrude_handle);
          cpd->new_point = cpd->acted = cpd->link_handles = true;
        }
        else if (extrude_point) {
          extrude_points_from_selected_vertices(
              &vc, obedit, event, extrude_center, extrude_handle);
          cpd->new_point = cpd->acted = cpd->link_handles = true;
        }
      }
    }
    else if (event->val == KM_RELEASE) {
      if (delete_point && !cpd->new_point && !cpd->dragging) {
        if (ED_curve_editnurb_select_pick_thresholded(
                C, event->mval, sel_dist_mul, false, false, false)) {
          cpd->acted = delete_point_under_mouse(&vc, event, sel_dist_mul);
        }
      }

      if (!cpd->acted && close_spline && close_spline_opts == ON_CLICK && cpd->found_point &&
          !cpd->dragging) {
        if (cpd->nu && !(cpd->nu->flagu & CU_NURB_CYCLIC)) {
          copy_v2_v2_int(vc.mval, event->mval);
          cpd->acted = make_cyclic_if_endpoints(cpd->nu, cpd->bezt, cpd->bp, &vc, C, sel_dist_mul);
        }
      }

      if (!cpd->acted && (insert_point || extrude_point) && cpd->spline_nearby) {
        if (!cpd->dragging) {
          if (insert_point) {
            insert_point_to_segment(event, vc.obedit->data, &nu, sel_dist_mul, &vc);
            cpd->new_point = true;
            cpd->acted = true;
          }
          else if (extrude_point) {
            extrude_points_from_selected_vertices(
                &vc, obedit, event, extrude_center, extrude_handle);
            cpd->acted = true;
          }
        }
      }

      if (!cpd->acted && toggle_vector) {
        short bezt_idx;
        get_closest_vertex_to_point_in_nurbs(
            nurbs, &nu, &bezt, &bp, &bezt_idx, mval_fl, sel_dist_mul, &vc);
        if (bezt) {
          if (bezt_idx == 1 && cycle_handle_type) {
            cycle_handles(bezt);
            cpd->acted = true;
          }
          else {
            toggle_handle_types(bezt, bezt_idx, cpd);
          }

          if (nu && nu->type == CU_BEZIER) {
            BKE_nurb_handles_calc(nu);
          }
        }
      }

      if (!cpd->selection_made) {
        if (!cpd->acted && select_multi) {
          short bezt_idx;
          get_closest_vertex_to_point_in_nurbs(
              nurbs, &nu, &bezt, &bp, &bezt_idx, mval_fl, sel_dist_mul, &vc);
          if (bezt) {
            toggle_select_bezt(bezt, bezt_idx);
            cu->actnu = get_nurb_index(nurbs, nu);
          }
          else if (bp) {
            toggle_select_bp(bp);
            cu->actnu = get_nurb_index(nurbs, nu);
          }
          else {
            ED_curve_deselect_all(cu->editnurb);
          }
        }
        else if (!cpd->acted && select_point) {
          ED_curve_editnurb_select_pick_thresholded(
              C, event->mval, sel_dist_mul, false, false, false);
        }
      }

      if (cpd->msd != NULL) {
        MEM_freeN(cpd->msd);
      }
      MEM_freeN(cpd);
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
  prop = RNA_def_float(ot->srna,
                       "sel_dist_mul",
                       0.2f,
                       0.0f,
                       2.0f,
                       "Select Distance",
                       "A multiplier on the default click distance",
                       0.1f,
                       1.5f);
  prop = RNA_def_enum(ot->srna,
                      "free_toggle",
                      prop_extra_key_types,
                      NONE,
                      "Free-Align Toggle",
                      "Toggle between free and align handles");
  prop = RNA_def_enum(ot->srna,
                      "adj_handle",
                      prop_extra_key_types,
                      NONE,
                      "Move Adjacent Handle",
                      "Move the closer handle of the adjacent vertex");
  prop = RNA_def_enum(ot->srna,
                      "move_entire",
                      prop_extra_key_types,
                      NONE,
                      "Move Entire",
                      "Move the whole point using handles");
  prop = RNA_def_enum(ot->srna,
                      "link_handles",
                      prop_extra_key_types,
                      NONE,
                      "Link Handles",
                      "Mirror the movement of one handle onto the other");
  prop = RNA_def_enum(ot->srna,
                      "lock_angle",
                      prop_extra_key_types,
                      NONE,
                      "Lock Handle Angle",
                      "Move the handle along its current angle");
  prop = RNA_def_boolean(ot->srna,
                         "extrude_point",
                         false,
                         "Extrude Point",
                         "Add a point connected to the last selected point");
  prop = RNA_def_boolean(ot->srna,
                         "extrude_center",
                         false,
                         "Extrude Internal",
                         "Allow extruding points from internal points");
  prop = RNA_def_enum(ot->srna,
                      "extrude_handle",
                      prop_handle_types,
                      HD_VECT,
                      "Extrude Handle Type",
                      "Mirror the movement of one handle onto the other");
  prop = RNA_def_boolean(ot->srna, "new_spline", false, "New Spline", "Create a new spline");
  prop = RNA_def_boolean(
      ot->srna, "delete_point", false, "Delete Point", "Delete an existing point");
  prop = RNA_def_boolean(
      ot->srna, "insert_point", false, "Insert Point", "Insert Point into a curve segment");
  prop = RNA_def_boolean(
      ot->srna, "move_segment", false, "Move Segment", "Delete an existing point");
  prop = RNA_def_boolean(
      ot->srna, "select_point", false, "Select Point", "Select a point or its handles");
  prop = RNA_def_boolean(
      ot->srna, "select_multi", false, "Select Multiple", "Select multiple points");
  prop = RNA_def_boolean(
      ot->srna, "move_point", false, "Move Point", "Move a point or its handles");
  prop = RNA_def_boolean(ot->srna,
                         "close_spline",
                         true,
                         "Close Spline",
                         "Make a spline cyclic by clicking endpoints");
  prop = RNA_def_enum(ot->srna,
                      "close_spline_opts",
                      prop_close_spline_opts,
                      OFF,
                      "Close Spline Method",
                      "The condition for close spline to activate");
  prop = RNA_def_boolean(
      ot->srna, "toggle_vector", false, "Toggle Vector", "Toggle between Vector and Auto handles");
  prop = RNA_def_boolean(ot->srna,
                         "cycle_handle_type",
                         false,
                         "Cycle Handle Type",
                         "Cycle between all four handle types");
}
