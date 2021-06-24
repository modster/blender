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
#include "DNA_object_types.h"
#include "DNA_scene_types.h"

#include "MEM_guardedalloc.h"

#include "BLI_blenlib.h"
#include "BLI_linklist.h"
#include "BLI_math.h"
#include "BLI_mempool.h"

#include "BKE_context.h"
#include "BKE_curve.h"
#include "BKE_fcurve.h"
#include "BKE_layer.h"
#include "BKE_report.h"

#include "DEG_depsgraph.h"

#include "WM_api.h"
#include "WM_toolsystem.h"
#include "WM_types.h"

#include "ED_curve.h"
#include "ED_object.h"
#include "ED_outliner.h"
#include "ED_screen.h"
#include "ED_select_utils.h"
#include "ED_space_api.h"
#include "ED_view3d.h"

#include "GPU_batch.h"
#include "GPU_batch_presets.h"
#include "GPU_immediate.h"
#include "GPU_immediate_util.h"
#include "GPU_matrix.h"
#include "GPU_state.h"

#include "BKE_object.h"
#include "BKE_paint.h"

#include "curve_intern.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "RNA_access.h"
#include "RNA_define.h"
#include "RNA_enum_types.h"

static void mouse_location_to_worldspace(const int *mouse_loc,
                                         const float *depth,
                                         const ViewContext *vc,
                                         float r_location[3])
{
  mul_v3_m4v3(r_location, vc->obedit->obmat, depth);
  ED_view3d_win_to_3d_int(vc->v3d, vc->region, r_location, mouse_loc, r_location);
}

static void move_bezt_handles_to_mouse(BezTriple *bezt,
                                       const bool is_end_point,
                                       const wmEvent *event,
                                       const ViewContext *vc)
{
  if (bezt->h1 == HD_VECT && bezt->h2 == HD_VECT) {
    bezt->h1 = HD_ALIGN;
    bezt->h2 = HD_ALIGN;
  }

  /* Obtain world space mouse location. */
  float location[3];
  mouse_location_to_worldspace(event->mval, bezt->vec[1], vc, location);

  /* If the new point is the last point of the curve, move the second handle. */
  if (is_end_point) {
    /* Set handle 2 location. */
    copy_v3_v3(bezt->vec[2], location);

    /* Set handle 1 location if handle not of type FREE */
    if (bezt->h2 != HD_FREE) {
      mul_v3_fl(location, -1);
      madd_v3_v3v3fl(bezt->vec[0], location, bezt->vec[1], 2);
    }
  }
  /* Else move the first handle. */
  else {
    /* Set handle 1 location. */
    copy_v3_v3(bezt->vec[0], location);

    /* Set handle 2 location if handle not of type FREE */
    if (bezt->h1 != HD_FREE) {
      mul_v3_fl(location, -1);
      madd_v3_v3v3fl(bezt->vec[2], location, bezt->vec[1], 2);
    }
  }
}

static void move_bezt_to_location(BezTriple *bezt, const float location[3])
{
  float change[3];
  sub_v3_v3v3(change, location, bezt->vec[1]);
  add_v3_v3(bezt->vec[0], change);
  copy_v3_v3(bezt->vec[1], location);
  add_v3_v3(bezt->vec[2], change);
}

static void free_up_selected_handles_for_movement(BezTriple *bezt)
{
  if (bezt->f1) {
    if (bezt->h1 == HD_VECT) {
      bezt->h1 = HD_FREE;
    }
    if (bezt->h1 == HD_AUTO) {
      bezt->h1 = HD_ALIGN;
      bezt->h2 = HD_ALIGN;
    }
  }
  else {
    if (bezt->h2 == HD_VECT) {
      bezt->h2 = HD_FREE;
    }
    if (bezt->h2 == HD_AUTO) {
      bezt->h1 = HD_ALIGN;
      bezt->h2 = HD_ALIGN;
    }
  }
}

static void delete_bezt_from_nurb(BezTriple *bezt, Nurb *nu)
{
  int index = BKE_curve_nurb_vert_index_get(nu, bezt);
  nu->pntsu -= 1;
  BezTriple *bezt1 = (BezTriple *)MEM_mallocN(nu->pntsu * sizeof(BezTriple), "NewBeztCurve");
  memcpy(bezt1, nu->bezt, index * sizeof(BezTriple));
  memcpy(bezt1 + index, nu->bezt + index + 1, (nu->pntsu - index) * sizeof(BezTriple));

  MEM_freeN(nu->bezt);
  nu->bezt = bezt1;
}

static void delete_bp_from_nurb(BPoint *bp, Nurb *nu)
{
  int index = BKE_curve_nurb_vert_index_get(nu, bp);
  nu->pntsu -= 1;
  BPoint *bp1 = (BPoint *)MEM_mallocN(nu->pntsu * sizeof(BPoint), "NewBpCurve");
  memcpy(bp1, nu->bp, index * sizeof(BPoint));
  memcpy(bp1 + index, nu->bp + index + 1, (nu->pntsu - index) * sizeof(BPoint));

  MEM_freeN(nu->bp);
  nu->bp = bp1;
}

static float get_view_zoom(const float *depth, const ViewContext *vc)
{
  int p1[2] = {0, 0};
  int p2[2] = {100, 0};
  float p1_3d[3], p2_3d[3];
  mouse_location_to_worldspace(p1, depth, vc, p1_3d);
  mouse_location_to_worldspace(p2, depth, vc, p2_3d);
  return 10 / len_v2v2(p1_3d, p2_3d);
}

static bool get_closest_point_on_edge(float *point,
                                      const float pos[2],
                                      const float pos1[3],
                                      const float pos2[3],
                                      const ViewContext *vc,
                                      float *factor)
{
  float pos1_2d[2], pos2_2d[2], vec1[2], vec2[2], vec3[2];

  /* Get screen space coordinates of points. */
  ED_view3d_project_float_object(
      vc->region, pos1, pos1_2d, V3D_PROJ_RET_CLIP_BB | V3D_PROJ_RET_CLIP_WIN);
  ED_view3d_project_float_object(
      vc->region, pos2, pos2_2d, V3D_PROJ_RET_CLIP_BB | V3D_PROJ_RET_CLIP_WIN);

  /* Obtain the vectors of each side. */
  sub_v2_v2v2(vec1, pos, pos1_2d);
  sub_v2_v2v2(vec2, pos2_2d, pos);
  sub_v2_v2v2(vec3, pos2_2d, pos1_2d);

  float dot1 = dot_v2v2(vec1, vec3);
  float dot2 = dot_v2v2(vec2, vec3);

  /* Compare the dot products to identify if both angles are optuse/acute or
  opposite to each other. If they're the same, that indicates that there is a
  perpendicular line from the mouse to the line.*/
  if ((dot1 > 0) == (dot2 > 0)) {
    float len_vec3_sq = len_squared_v2(vec3);
    *factor = 1 - dot2 / len_vec3_sq;

    float pos_dif[3];
    sub_v3_v3v3(pos_dif, pos2, pos1);
    madd_v3_v3v3fl(point, pos1, pos_dif, *factor);
    return true;
  }
  if (len_manhattan_v2(vec1) < len_manhattan_v2(vec2)) {
    copy_v3_v3(point, pos1);
    return false;
  }
  copy_v3_v3(point, pos2);
  return false;
}

static BezTriple *get_closest_bezt_to_point(Nurb *nu, const float point[2], const ViewContext *vc)
{
  float min_distance = 10000;

  BezTriple *closest = NULL;
  for (int i = 0; i < nu->pntsu; i++) {
    BezTriple *bezt = &nu->bezt[i];
    float bezt_vec[2];
    ED_view3d_project_float_object(
        vc->region, bezt->vec[1], bezt_vec, V3D_PROJ_RET_CLIP_BB | V3D_PROJ_RET_CLIP_WIN);
    float distance = len_manhattan_v2v2(bezt_vec, point);
    if (distance < min_distance) {
      min_distance = distance;
      closest = bezt;
    }
  }
  if (closest) {
    float threshold_distance = get_view_zoom(closest->vec[1], vc);
    if (min_distance < threshold_distance) {
      return closest;
    }
  }
  return NULL;
}

static BPoint *get_closest_bp_to_point(Nurb *nu, const float point[2], const ViewContext *vc)
{
  float min_distance = 10000;
  float temp[2];
  copy_v2_v2(temp, point);
  BPoint *closest = NULL;
  for (int i = 0; i < nu->pntsu; i++) {
    BPoint *bp = &nu->bp[i];
    float bp_vec[2];
    ED_view3d_project_float_object(
        vc->region, bp->vec, bp_vec, V3D_PROJ_RET_CLIP_BB | V3D_PROJ_RET_CLIP_WIN);
    float distance = len_manhattan_v2v2(bp_vec, point);
    if (distance < min_distance) {
      min_distance = distance;
      closest = bp;
    }
  }
  if (closest) {
    float threshold_distance = get_view_zoom(closest->vec, vc);
    if (min_distance < threshold_distance) {
      return closest;
    }
  }
  return NULL;
}

static void select_and_get_point(ViewContext *vc,
                                 Nurb **nu,
                                 BezTriple **bezt,
                                 BPoint **bp,
                                 const int point[2],
                                 const bool is_start)
{
  short hand;
  BezTriple *bezt1 = NULL;
  BPoint *bp1 = NULL;
  Base *basact1 = NULL;
  Nurb *nu1 = NULL;
  Curve *cu = vc->obedit->data;
  copy_v2_v2_int(vc->mval, point);
  if (is_start) {
    ED_curve_pick_vert(vc, 1, &nu1, &bezt1, &bp1, &hand, &basact1);
  }
  else {
    ED_curve_nurb_vert_selected_find(cu, vc->v3d, &nu1, &bezt1, &bp1);
  }
  *bezt = bezt1;
  *bp = bp1;
  *nu = nu1;
}

static void calculate_new_bezier_point(const float *point_prev,
                                       float *handle_prev,
                                       float *new_left_handle,
                                       float *new_right_handle,
                                       float *handle_next,
                                       const float *point_next,
                                       const float parameter)
{
  const float center_point[3];
  interp_v3_v3v3(center_point, handle_prev, handle_next, parameter);
  interp_v3_v3v3(handle_prev, point_prev, handle_prev, parameter);
  interp_v3_v3v3(handle_next, handle_next, point_next, parameter);
  interp_v3_v3v3(new_left_handle, handle_prev, center_point, parameter);
  interp_v3_v3v3(new_right_handle, center_point, handle_next, parameter);
}

static void update_data_if_nearest_point_in_segment(BezTriple *bezt1,
                                                    BezTriple *bezt2,
                                                    Nurb *nu,
                                                    int index,
                                                    ViewContext *vc,
                                                    float screen_co[2],
                                                    void *op_data)
{
  struct temp_bezt_data {
    /* Index of the last bez triple before the cut. */
    int bezt_index;
    /* Nurb to which the cut belongs to. */
    Nurb *nurb;
    /* Minimum distance to curve from mouse location. */
    float min_dist;
    /* Ratio at which the new point divides the curve segment. */
    float parameter;
    /* Whether the cut has any vertices before/after it. */
    bool has_prev, has_next;
    /* Locations of adjacent vertices. */
    float prev_loc[3], cut_loc[3], next_loc[3];
    /* Mouse location as floats. */
    float mval[2];
  } temp_bezt_data;
  struct temp_bezt_data *data = op_data;

  float resolu = nu->resolu;
  float *points = MEM_mallocN(sizeof(float[3]) * (resolu + 1), "makeCut_bezier");

  /* Calculate all points on curve. TODO: Get existing . */
  for (int j = 0; j < 3; j++) {
    BKE_curve_forward_diff_bezier(bezt1->vec[1][j],
                                  bezt1->vec[2][j],
                                  bezt2->vec[0][j],
                                  bezt2->vec[1][j],
                                  points + j,
                                  resolu,
                                  sizeof(float[3]));
  }

  /* Calculate angle for middle points */
  for (int k = 0; k <= resolu; k++) {
    /* Convert point to screen coordinates */
    bool check = ED_view3d_project_float_object(vc->region,
                                                points + 3 * k,
                                                screen_co,
                                                V3D_PROJ_RET_CLIP_BB | V3D_PROJ_RET_CLIP_WIN) ==
                 V3D_PROJ_RET_OK;

    if (check) {
      float distance = len_manhattan_v2v2(screen_co, data->mval);
      if (distance < data->min_dist) {
        data->min_dist = distance;
        data->nurb = nu;
        data->bezt_index = index;
        data->parameter = ((float)k) / resolu;

        copy_v3_v3(data->cut_loc, points + 3 * k);

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

enum {
  PEN_MODAL_CANCEL = 1,
  PEN_MODAL_FREE_MOVE_HANDLE,
};

wmKeyMap *curve_pen_modal_keymap(wmKeyConfig *keyconf)
{
  static const EnumPropertyItem modal_items[] = {
      {PEN_MODAL_CANCEL, "CANCEL", 0, "Cancel", "Cancel pen"},
      {PEN_MODAL_FREE_MOVE_HANDLE,
       "FREE_MOVE_HANDLE",
       0,
       "Free Move handle",
       "Move handle of newly added point freely"},
      {0, NULL, 0, NULL, NULL},
  };

  wmKeyMap *keymap = WM_modalkeymap_find(keyconf, "Curve Pen Modal Map");

  /* This function is called for each spacetype, only needs to add map once */
  if (keymap && keymap->modal_items) {
    return NULL;
  }

  keymap = WM_modalkeymap_ensure(keyconf, "Curve Pen Modal Map", modal_items);

  WM_modalkeymap_assign(keymap, "CURVE_OT_pen");

  return keymap;
}

static int curve_pen_modal(bContext *C, wmOperator *op, const wmEvent *event)
{
  bool extend = RNA_boolean_get(op->ptr, "extend");
  bool deselect = RNA_boolean_get(op->ptr, "deselect");
  bool toggle = RNA_boolean_get(op->ptr, "toggle");
  Depsgraph *depsgraph = CTX_data_ensure_evaluated_depsgraph(C);
  ViewContext vc;
  Object *obedit = CTX_data_edit_object(C);

  ED_view3d_viewcontext_init(C, &vc, depsgraph);
  Curve *cu = vc.obedit->data;

  BezTriple *bezt = NULL;
  BPoint *bp = NULL;
  Nurb *nu = NULL;

  bool retval = false;

  view3d_operator_needs_opengl(C);
  BKE_object_update_select_id(CTX_data_main(C));

  int ret = OPERATOR_RUNNING_MODAL;
  bool dragging = RNA_boolean_get(op->ptr, "dragging");

  bool picked = false;
  if (event->type == EVT_MODAL_MAP) {
    if (event->val == PEN_MODAL_FREE_MOVE_HANDLE) {
      select_and_get_point(&vc, &nu, &bezt, &bp, event->mval, event->prevval != KM_PRESS);
      picked = true;

      if (bezt) {
        bezt->h1 = bezt->h2 = HD_FREE;
      }
    }
  }

  if (ELEM(event->type, MOUSEMOVE, INBETWEEN_MOUSEMOVE)) {
    if (!dragging && WM_event_drag_test(event, &event->prevclickx) && event->val == KM_PRESS) {
      RNA_boolean_set(op->ptr, "dragging", true);
      dragging = true;
    }
    if (dragging) {
      /* Move handle point with mouse cursor if dragging a new control point. */
      if (RNA_boolean_get(op->ptr, "new")) {
        if (!picked) {
          select_and_get_point(&vc, &nu, &bezt, &bp, event->mval, event->prevval != KM_PRESS);
        }
        if (bezt) {
          move_bezt_handles_to_mouse(bezt, nu->bezt + nu->pntsu - 1 == bezt, event, &vc);

          BKE_nurb_handles_calc(nu);
        }
      }
      /* Move entire control point with mouse cursor if dragging an existing control point. */
      else {
        select_and_get_point(&vc, &nu, &bezt, &bp, event->mval, event->prevval != KM_PRESS);

        if (bezt) {
          /* Get mouse location in 3D space. */
          float location[3];
          mouse_location_to_worldspace(event->mval, bezt->vec[1], &vc, location);

          /* Move entire BezTriple if center point is dragged. */
          if (bezt->f2) {
            move_bezt_to_location(bezt, location);
          }
          /* Move handle separately if only a handle is dragged. */
          else {
            free_up_selected_handles_for_movement(bezt);
            if (bezt->f1) {
              copy_v3_v3(bezt->vec[0], location);
            }
            else {
              copy_v3_v3(bezt->vec[2], location);
            }
          }

          /* Other handle automatically calculated */
          BKE_nurb_handles_calc(nu);
        }
        else if (bp) {
          /* Get mouse location in 3D space. */
          float location[3];
          mouse_location_to_worldspace(event->mval, bp->vec, &vc, location);

          copy_v3_v3(bp->vec, location);

          BKE_nurb_handles_calc(nu);
        }
      }
    }
  }
  else if (ELEM(event->type, LEFTMOUSE)) {
    if (event->val == KM_PRESS) {
      retval = ED_curve_editnurb_select_pick(C, event->mval, extend, deselect, toggle);
      RNA_boolean_set(op->ptr, "new", !retval);
      bool cut_or_delete = RNA_boolean_get(op->ptr, "cut_or_delete");

      /* Check if point underneath mouse. Get point if any. */
      if (retval) {
        if (cut_or_delete) {
          /* Delete retrieved control point. */
          ListBase *nurbs = BKE_curve_editNurbs_get(cu);
          float mouse_point[2] = {(float)event->mval[0], (float)event->mval[1]};

          for (nu = nurbs->first; nu; nu = nu->next) {
            if (nu->type == CU_BEZIER) {
              bezt = get_closest_bezt_to_point(nu, mouse_point, &vc);
              if (bezt && nu) {
                delete_bezt_from_nurb(bezt, nu);
              }
            }
            else if (nu->type == CU_NURBS) {
              bp = get_closest_bp_to_point(nu, mouse_point, &vc);
              if (bp && nu) {
                delete_bp_from_nurb(bp, nu);
              }
            }
          }

          cu->actvert = CU_ACT_NONE;
          if (nu) {
            BKE_nurb_handles_calc(nu);
          }
        }
      }
      else {
        if (cut_or_delete) {
          /* If curve segment is nearby, add control point at the snapped point
          between the adjacent control points in the curve data structure. */
          EditNurb *editnurb = cu->editnurb;

          /* Data structure to keep track of details about the cut location */
          struct {
            /* Index of the last bez triple before the cut. */
            int bezt_index;
            /* Nurb to which the cut belongs to. */
            Nurb *nurb;
            /* Minimum distance to curve from mouse location. */
            float min_dist;
            /* Ratio at which the new point divides the curve segment. */
            float parameter;
            /* Whether the cut has any vertices before/after it. */
            bool has_prev, has_next;
            /* Locations of adjacent vertices. */
            float prev_loc[3], cut_loc[3], next_loc[3];
            /* Mouse location as floats. */
            float mval[2];
          } data = {.bezt_index = 0,
                    .min_dist = 10000,
                    .parameter = 0.5f,
                    .has_prev = false,
                    .has_next = false};

          data.mval[0] = event->mval[0];
          data.mval[1] = event->mval[1];

          ListBase *nurbs = BKE_curve_editNurbs_get(cu);

          for (nu = nurbs->first; nu; nu = nu->next) {
            if (nu->type == CU_BEZIER) {
              float screen_co[2];
              if (data.nurb == NULL) {
                ED_view3d_project_float_object(vc.region,
                                               nu->bezt->vec[1],
                                               screen_co,
                                               V3D_PROJ_RET_CLIP_BB | V3D_PROJ_RET_CLIP_WIN);

                data.nurb = nu;
                data.bezt_index = 0;
                data.min_dist = len_manhattan_v2v2(screen_co, data.mval);
                copy_v3_v3(data.cut_loc, nu->bezt->vec[1]);
              }
              int i;
              for (i = 0; i < nu->pntsu - 1; i++) {
                bezt = &nu->bezt[i];
                update_data_if_nearest_point_in_segment(
                    bezt, bezt + 1, nu, i, &vc, screen_co, &data);
              }

              if (nu->flagu & CU_NURB_CYCLIC) {
                update_data_if_nearest_point_in_segment(
                    bezt + 1, nu->bezt, nu, i, &vc, screen_co, &data);
              }
            }
          }
          float threshold_distance = get_view_zoom(data.cut_loc, &vc);
          /* If the minimum distance found < threshold distance, make cut. */
          if (data.min_dist < threshold_distance) {
            nu = data.nurb;
            if (nu && nu->bezt) {
              bool found_min = false;
              float point[3];
              float factor;
              if (data.has_prev) {
                found_min = get_closest_point_on_edge(
                    point, data.mval, data.cut_loc, data.prev_loc, &vc, &factor);
                factor = -factor;
              }
              if (!found_min && data.has_next) {
                found_min = get_closest_point_on_edge(
                    point, data.mval, data.cut_loc, data.next_loc, &vc, &factor);
              }
              if (found_min) {
                float point_2d[2];
                ED_view3d_project_float_object(
                    vc.region, point, point_2d, V3D_PROJ_RET_CLIP_BB | V3D_PROJ_RET_CLIP_WIN);
                float dist = len_manhattan_v2v2(point_2d, data.mval);
                data.min_dist = dist;
                data.parameter += factor / nu->resolu;
                copy_v3_v3(data.cut_loc, point);
              }

              BezTriple *bezt1 = (BezTriple *)MEM_mallocN((nu->pntsu + 1) * sizeof(BezTriple),
                                                          "delNurb");
              int index = data.bezt_index + 1;
              /* Copy all control points before the cut to the new memory. */
              memcpy(bezt1, nu->bezt, index * sizeof(BezTriple));
              BezTriple *new_bezt = bezt1 + index;

              /* Duplicate control point after the cut. */
              memcpy(new_bezt, new_bezt - 1, sizeof(BezTriple));
              copy_v3_v3(new_bezt->vec[1], data.cut_loc);

              if (index < nu->pntsu) {
                /* Copy all control points after the cut to the new memory. */
                memcpy(
                    bezt1 + index + 1, nu->bezt + index, (nu->pntsu - index) * sizeof(BezTriple));
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

              calculate_new_bezier_point((new_bezt - 1)->vec[1],
                                         (new_bezt - 1)->vec[2],
                                         new_bezt->vec[0],
                                         new_bezt->vec[2],
                                         next_bezt->vec[0],
                                         next_bezt->vec[1],
                                         data.parameter);

              MEM_freeN(nu->bezt);
              nu->bezt = bezt1;
              ED_curve_deselect_all(editnurb);
              BKE_nurb_handles_calc(nu);
              new_bezt->f1 = new_bezt->f2 = new_bezt->f3 = 1;
            }
          }
        }
        else {
          /* Create new point under the mouse cursor. Set handle types as vector.
          If an end point of a spline is selected, set the new point as the
          new end point of the spline. */
          float location[3];

          ED_curve_nurb_vert_selected_find(cu, vc.v3d, &nu, &bezt, &bp);

          if (bezt) {
            mul_v3_m4v3(location, vc.obedit->obmat, bezt->vec[1]);
          }
          else if (bp) {
            mul_v3_m4v3(location, vc.obedit->obmat, bp->vec);
          }
          else {
            copy_v3_v3(location, vc.scene->cursor.location);
          }

          ED_view3d_win_to_3d_int(vc.v3d, vc.region, location, event->mval, location);
          EditNurb *editnurb = cu->editnurb;
          ed_editcurve_addvert(cu, editnurb, vc.v3d, location);
          ED_curve_nurb_vert_selected_find(cu, vc.v3d, &nu, &bezt, &bp);
          if (bezt) {
            bezt->h1 = HD_VECT;
            bezt->h2 = HD_VECT;
          }
        }
      }
    }
    if (event->val == KM_RELEASE) {
      if (dragging) {
        RNA_boolean_set(op->ptr, "dragging", false);
      }
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
  // return view3d_select_invoke(C, op, event);
  return ret;
}

void CURVE_OT_pen(wmOperatorType *ot)
{
  /* identifiers */
  ot->name = "Curve Pen";
  ot->idname = "CURVE_OT_pen";
  ot->description = "Edit curves with less shortcuts";

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
  RNA_def_property_flag(prop, PROP_SKIP_SAVE);
  prop = RNA_def_boolean(
      ot->srna, "new", 0, "New Point Drag", "The point was added with the press before drag");
  RNA_def_property_flag(prop, PROP_SKIP_SAVE);
  prop = RNA_def_boolean(ot->srna, "wait_for_input", true, "Wait for Input", "");
  RNA_def_property_flag(prop, PROP_HIDDEN | PROP_SKIP_SAVE);
  prop = RNA_def_boolean(
      ot->srna, "cut_or_delete", true, "Whether cut or delete key bindings are pressed", "");
  RNA_def_property_flag(prop, PROP_HIDDEN | PROP_SKIP_SAVE);
}
