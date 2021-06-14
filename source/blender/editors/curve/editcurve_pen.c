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

static bool ed_editcurve_extrude(Curve *cu, EditNurb *editnurb, View3D *v3d)
{
  bool changed = false;

  Nurb *cu_actnu;
  union {
    BezTriple *bezt;
    BPoint *bp;
    void *p;
  } cu_actvert;

  if (BLI_listbase_is_empty(&editnurb->nurbs)) {
    return changed;
  }

  BKE_curve_nurb_vert_active_get(cu, &cu_actnu, &cu_actvert.p);
  int act_offset = 0;

  LISTBASE_FOREACH (Nurb *, nu, &editnurb->nurbs) {
    BLI_assert(nu->pntsu > 0);
    int i;
    int pnt_len = nu->pntsu;
    int new_points = 0;
    int offset = 0;
    bool is_prev_selected = false;
    bool duplic_first = false;
    bool duplic_last = false;
    if (nu->type == CU_BEZIER) {
      BezTriple *bezt, *bezt_prev = NULL;
      BezTriple bezt_stack;
      bool is_cyclic = false;
      if (pnt_len == 1) {
        /* Single point extrusion.
         * Keep `is_prev_selected` false to force extrude. */
        bezt_prev = &nu->bezt[0];
      }
      else if (nu->flagu & CU_NURB_CYCLIC) {
        is_cyclic = true;
        bezt_prev = &nu->bezt[pnt_len - 1];
        is_prev_selected = BEZT_ISSEL_ANY_HIDDENHANDLES(v3d, bezt_prev);
      }
      else {
        duplic_first = BEZT_ISSEL_ANY_HIDDENHANDLES(v3d, &nu->bezt[0]) &&
                       BEZT_ISSEL_ANY_HIDDENHANDLES(v3d, &nu->bezt[1]);

        duplic_last = BEZT_ISSEL_ANY_HIDDENHANDLES(v3d, &nu->bezt[pnt_len - 2]) &&
                      BEZT_ISSEL_ANY_HIDDENHANDLES(v3d, &nu->bezt[pnt_len - 1]);

        if (duplic_first) {
          bezt_stack = nu->bezt[0];
          BEZT_DESEL_ALL(&bezt_stack);
          bezt_prev = &bezt_stack;
        }
        if (duplic_last) {
          new_points++;
        }
      }
      i = pnt_len;
      for (bezt = &nu->bezt[0]; i--; bezt++) {
        bool is_selected = BEZT_ISSEL_ANY_HIDDENHANDLES(v3d, bezt);
        if (bezt_prev && is_prev_selected != is_selected) {
          new_points++;
        }
        if (bezt == cu_actvert.bezt) {
          act_offset = new_points;
        }
        bezt_prev = bezt;
        is_prev_selected = is_selected;
      }

      if (new_points) {
        if (pnt_len == 1) {
          /* Single point extrusion.
           * Set `is_prev_selected` as false to force extrude. */
          BLI_assert(bezt_prev == &nu->bezt[0]);
          is_prev_selected = false;
        }
        else if (is_cyclic) {
          BLI_assert(bezt_prev == &nu->bezt[pnt_len - 1]);
          BLI_assert(is_prev_selected == BEZT_ISSEL_ANY_HIDDENHANDLES(v3d, bezt_prev));
        }
        else if (duplic_first) {
          bezt_prev = &bezt_stack;
          is_prev_selected = false;
        }
        else {
          bezt_prev = NULL;
        }
        BezTriple *bezt_src, *bezt_dst, *bezt_src_iter, *bezt_dst_iter;
        const int new_len = pnt_len + new_points;

        bezt_src = nu->bezt;
        bezt_dst = MEM_mallocN(new_len * sizeof(BezTriple), __func__);
        bezt_src_iter = &bezt_src[0];
        bezt_dst_iter = &bezt_dst[0];
        i = 0;
        for (bezt = &nu->bezt[0]; i < pnt_len; i++, bezt++) {
          bool is_selected = BEZT_ISSEL_ANY_HIDDENHANDLES(v3d, bezt);
          /* While this gets de-selected, selecting here ensures newly created verts are selected.
           * without this, the vertices are copied but only the handles are transformed.
           * which seems buggy from a user perspective. */
          if (is_selected) {
            bezt->f2 |= SELECT;
          }
          if (bezt_prev && is_prev_selected != is_selected) {
            int count = i - offset + 1;
            if (is_prev_selected) {
              ED_curve_beztcpy(editnurb, bezt_dst_iter, bezt_src_iter, count - 1);
              ED_curve_beztcpy(editnurb, &bezt_dst_iter[count - 1], bezt_prev, 1);
            }
            else {
              ED_curve_beztcpy(editnurb, bezt_dst_iter, bezt_src_iter, count);
            }
            ED_curve_beztcpy(editnurb, &bezt_dst_iter[count], bezt, 1);
            BEZT_DESEL_ALL(&bezt_dst_iter[count - 1]);

            bezt_dst_iter += count + 1;
            bezt_src_iter += count;
            offset = i + 1;
          }
          bezt_prev = bezt;
          is_prev_selected = is_selected;
        }

        int remain = pnt_len - offset;
        if (remain) {
          ED_curve_beztcpy(editnurb, bezt_dst_iter, bezt_src_iter, remain);
        }

        if (duplic_last) {
          ED_curve_beztcpy(editnurb, &bezt_dst[new_len - 1], &bezt_src[pnt_len - 1], 1);
          BEZT_DESEL_ALL(&bezt_dst[new_len - 1]);
        }

        MEM_freeN(nu->bezt);
        nu->bezt = bezt_dst;
        nu->pntsu += new_points;
        changed = true;
      }
    }
    else {
      BPoint *bp, *bp_prev = NULL;
      BPoint bp_stack;
      if (pnt_len == 1) {
        /* Single point extrusion.
         * Reference a `prev_bp` to force extrude. */
        bp_prev = &nu->bp[0];
      }
      else {
        duplic_first = (nu->bp[0].f1 & SELECT) && (nu->bp[1].f1 & SELECT);
        duplic_last = (nu->bp[pnt_len - 2].f1 & SELECT) && (nu->bp[pnt_len - 1].f1 & SELECT);
        if (duplic_first) {
          bp_stack = nu->bp[0];
          bp_stack.f1 &= ~SELECT;
          bp_prev = &bp_stack;
        }
        if (duplic_last) {
          new_points++;
        }
      }

      i = pnt_len;
      for (bp = &nu->bp[0]; i--; bp++) {
        bool is_selected = (bp->f1 & SELECT) != 0;
        if (bp_prev && is_prev_selected != is_selected) {
          new_points++;
        }
        if (bp == cu_actvert.bp) {
          act_offset = new_points;
        }
        bp_prev = bp;
        is_prev_selected = is_selected;
      }

      if (new_points) {
        BPoint *bp_src, *bp_dst, *bp_src_iter, *bp_dst_iter;
        const int new_len = pnt_len + new_points;

        is_prev_selected = false;
        if (pnt_len == 1) {
          /* Single point extrusion.
           * Keep `is_prev_selected` false to force extrude. */
          BLI_assert(bp_prev == &nu->bp[0]);
        }
        else if (duplic_first) {
          bp_prev = &bp_stack;
          is_prev_selected = false;
        }
        else {
          bp_prev = NULL;
        }
        bp_src = nu->bp;
        bp_dst = MEM_mallocN(new_len * sizeof(BPoint), __func__);
        bp_src_iter = &bp_src[0];
        bp_dst_iter = &bp_dst[0];
        i = 0;
        for (bp = &nu->bp[0]; i < pnt_len; i++, bp++) {
          bool is_selected = (bp->f1 & SELECT) != 0;
          if (bp_prev && is_prev_selected != is_selected) {
            int count = i - offset + 1;
            if (is_prev_selected) {
              ED_curve_bpcpy(editnurb, bp_dst_iter, bp_src_iter, count - 1);
              ED_curve_bpcpy(editnurb, &bp_dst_iter[count - 1], bp_prev, 1);
            }
            else {
              ED_curve_bpcpy(editnurb, bp_dst_iter, bp_src_iter, count);
            }
            ED_curve_bpcpy(editnurb, &bp_dst_iter[count], bp, 1);
            bp_dst_iter[count - 1].f1 &= ~SELECT;

            bp_dst_iter += count + 1;
            bp_src_iter += count;
            offset = i + 1;
          }
          bp_prev = bp;
          is_prev_selected = is_selected;
        }

        int remain = pnt_len - offset;
        if (remain) {
          ED_curve_bpcpy(editnurb, bp_dst_iter, bp_src_iter, remain);
        }

        if (duplic_last) {
          ED_curve_bpcpy(editnurb, &bp_dst[new_len - 1], &bp_src[pnt_len - 1], 1);
          bp_dst[new_len - 1].f1 &= ~SELECT;
        }

        MEM_freeN(nu->bp);
        nu->bp = bp_dst;
        nu->pntsu += new_points;

        BKE_nurb_knot_calc_u(nu);
        changed = true;
      }
    }
  }

  cu->actvert += act_offset;

  return changed;
}

static int ed_editcurve_addvert(bContext *C, const float location_init[3])
{
  Object *obedit = CTX_data_edit_object(C);
  View3D *v3d = CTX_wm_view3d(C);
  Curve *cu = obedit->data;
  EditNurb *editnurb = cu->editnurb;
  float location[3];
  float center[3];
  float temp[3];
  uint verts_len;
  bool changed = false;

  zero_v3(center);
  verts_len = 0;

  LISTBASE_FOREACH (Nurb *, nu, &editnurb->nurbs) {
    int i;
    if (nu->type == CU_BEZIER) {
      BezTriple *bezt;

      for (i = 0, bezt = nu->bezt; i < nu->pntsu; i++, bezt++) {
        if (BEZT_ISSEL_ANY_HIDDENHANDLES(v3d, bezt)) {
          add_v3_v3(center, bezt->vec[1]);
          verts_len += 1;
        }
      }
    }
    else {
      BPoint *bp;

      for (i = 0, bp = nu->bp; i < nu->pntsu; i++, bp++) {
        if (bp->f1 & SELECT) {
          add_v3_v3(center, bp->vec);
          verts_len += 1;
        }
      }
    }
  }

  if (verts_len && ed_editcurve_extrude(cu, editnurb, v3d)) {
    float ofs[3];
    int i;

    mul_v3_fl(center, 1.0f / (float)verts_len);
    sub_v3_v3v3(ofs, location_init, center);

    if (CU_IS_2D(cu)) {
      ofs[2] = 0.0f;
    }

    LISTBASE_FOREACH (Nurb *, nu, &editnurb->nurbs) {
      if (nu->type == CU_BEZIER) {
        BezTriple *bezt;
        for (i = 0, bezt = nu->bezt; i < nu->pntsu; i++, bezt++) {
          if (BEZT_ISSEL_ANY_HIDDENHANDLES(v3d, bezt)) {
            add_v3_v3(bezt->vec[0], ofs);
            add_v3_v3(bezt->vec[1], ofs);
            add_v3_v3(bezt->vec[2], ofs);
            bezt->h1 = HD_VECT;
            bezt->h2 = HD_VECT;

            if (((nu->flagu & CU_NURB_CYCLIC) == 0) && (i == 0 || i == nu->pntsu - 1)) {
              BKE_nurb_handle_calc_simple_auto(nu, bezt);
            }
          }
        }

        BKE_nurb_handles_calc(nu);
      }
      else {
        BPoint *bp;

        for (i = 0, bp = nu->bp; i < nu->pntsu; i++, bp++) {
          if (bp->f1 & SELECT) {
            add_v3_v3(bp->vec, ofs);
          }
        }
      }
    }
    changed = true;
  }
  else {
    float location[3];

    copy_v3_v3(location, location_init);

    if (CU_IS_2D(cu)) {
      location[2] = 0.0f;
    }

    /* nothing selected: create a new curve */
    Nurb *nu = BKE_curve_nurb_active_get(cu);

    Nurb *nurb_new;
    if (!nu) {
      /* Bezier as default. */
      nurb_new = MEM_callocN(sizeof(Nurb), "BLI_editcurve_addvert new_bezt_nurb 2");
      nurb_new->type = CU_BEZIER;
      nurb_new->resolu = cu->resolu;
      nurb_new->orderu = 4;
      nurb_new->flag |= CU_SMOOTH;
      BKE_nurb_bezierPoints_add(nurb_new, 1);
    }
    else {
      /* Copy the active nurb settings. */
      nurb_new = BKE_nurb_copy(nu, 1, 1);
      if (nu->bezt) {
        memcpy(nurb_new->bezt, nu->bezt, sizeof(BezTriple));
      }
      else {
        memcpy(nurb_new->bp, nu->bp, sizeof(BPoint));
      }
    }

    if (nurb_new->type == CU_BEZIER) {
      BezTriple *bezt_new = nurb_new->bezt;

      BEZT_SEL_ALL(bezt_new);

      bezt_new->h1 = HD_AUTO;
      bezt_new->h2 = HD_AUTO;

      temp[0] = 1.0f;
      temp[1] = 0.0f;
      temp[2] = 0.0f;

      copy_v3_v3(bezt_new->vec[1], location);
      sub_v3_v3v3(bezt_new->vec[0], location, temp);
      add_v3_v3v3(bezt_new->vec[2], location, temp);
    }
    else {
      BPoint *bp_new = nurb_new->bp;

      bp_new->f1 |= SELECT;

      copy_v3_v3(bp_new->vec, location);

      BKE_nurb_knot_calc_u(nurb_new);
    }

    BLI_addtail(&editnurb->nurbs, nurb_new);
    changed = true;
  }

  return changed;
}

static void curve_edit_exit(wmOperator *op)
{
}

static void curve_edit_cancel(bContext *UNUSED(C), wmOperator *op)
{
  curve_edit_exit(op);
}

static float *mouse_location_to_worldspace(int *mouse_loc, float *depth, ViewContext vc)
{
  float location[3];
  mul_v3_m4v3(location, vc.obedit->obmat, depth);
  ED_view3d_win_to_3d_int(vc.v3d, vc.region, location, mouse_loc, location);
  return location;
}

static void move_bezt_handles_to_mouse(BezTriple *bezt,
                                       bool is_end_point,
                                       const wmEvent *event,
                                       ViewContext vc)
{
  if (bezt->h1 == HD_VECT && bezt->h2 == HD_VECT) {
    bezt->h1 = HD_ALIGN;
    bezt->h2 = HD_ALIGN;
  }

  /* Set handle type to free when moved with shift. */
  if (event->shift && bezt->h1 != HD_FREE) {
    bezt->h1 = HD_FREE;
    bezt->h2 = HD_FREE;
  }

  /* Obtain world space mouse location. */
  float *location = mouse_location_to_worldspace(event->mval, bezt->vec[1], vc);

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

static void move_bezt_to_location(BezTriple *bezt, float location[3])
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

static float *get_closest_point_on_edge(const float pos[2],
                                        const float pos1[3],
                                        const float pos2[3],
                                        ViewContext vc)
{
  float pos1_2d[2], pos2_2d[2], vec1[2], vec2[2], vec3[2];

  /* Get screen space coordinates of points. */
  ED_view3d_project_float_object(
      vc.region, pos1, pos1_2d, V3D_PROJ_RET_CLIP_BB | V3D_PROJ_RET_CLIP_WIN) == V3D_PROJ_RET_OK;
  ED_view3d_project_float_object(
      vc.region, pos2, pos2_2d, V3D_PROJ_RET_CLIP_BB | V3D_PROJ_RET_CLIP_WIN) == V3D_PROJ_RET_OK;

  /* Obtain the vectors of each side. */
  sub_v2_v2v2(vec1, pos, pos1_2d);
  sub_v2_v2v2(vec2, pos2_2d, pos);
  sub_v2_v2v2(vec3, pos2_2d, pos1_2d);

  float dot1 = dot_v2v2(vec1, vec3);
  float dot2 = dot_v2v2(vec2, vec3);
  float len_vec2_sq = len_squared_v2(vec2);

  /* Compare the dot products to identify if both angles are optuse/acute or
  opposite to each other. If they're the same, that indicates that there is a
  perpendicular line from the mouse to the line.*/
  if ((dot1 > 0) == (dot2 > 0)) {
    float len_vec1_sq = len_squared_v2(vec3);
    float perp_dist = sqrt(len_vec2_sq - dot2 * dot2 / len_vec1_sq);
    float factor = 1 - dot2 / len_vec1_sq;

    float pos_dif[3], intersect[3];
    sub_v3_v3v3(pos_dif, pos2, pos1);
    madd_v3_v3v3fl(intersect, pos1, pos_dif, factor);
    return intersect;
  }
  if (len_manhattan_v2(vec1) < len_manhattan_v2(vec2)) {
    return pos1;
  }
  return pos2;
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
  Base *basact = NULL;
  Nurb *nu;

  bool retval = false;
  int mval[2];

  RNA_int_get_array(op->ptr, "location", mval);
  view3d_operator_needs_opengl(C);
  BKE_object_update_select_id(CTX_data_main(C));

  int ret = OPERATOR_RUNNING_MODAL;
  bool dragging = RNA_boolean_get(op->ptr, "dragging");

  if (ELEM(event->type, MOUSEMOVE, INBETWEEN_MOUSEMOVE)) {
    if (!dragging && WM_event_drag_test(event, &event->prevclickx) && event->val == KM_PRESS) {
      RNA_boolean_set(op->ptr, "dragging", true);
      dragging = true;
    }
    if (dragging) {
      /* Move handle point with mouse cursor if dragging a new control point. */
      if (RNA_boolean_get(op->ptr, "new")) {
        short hand;

        copy_v2_v2_int(vc.mval, mval);
        /* Get pointer to new control point. */
        ED_curve_pick_vert(&vc, 1, &nu, &bezt, &bp, &hand, &basact);
        if (bezt) {
          move_bezt_handles_to_mouse(bezt, nu->bezt + nu->pntsu - 1 == bezt, event, vc);

          BKE_nurb_handles_calc(nu);
        }
      }
      /* Move entire control point with mouse cursor if dragging an existing control point. */
      else {
        short hand;

        mval[0] = event->prevx;
        mval[1] = event->prevy;
        copy_v2_v2_int(vc.mval, mval);

        /* Select point if the mouse was just clicked and get the selected point. */
        if (event->prevval != KM_PRESS) {
          ED_curve_pick_vert(&vc, 1, &nu, &bezt, &bp, &hand, &basact);
        }
        else {
          ED_curve_nurb_vert_selected_find(cu, vc.v3d, &nu, &bezt, &bp);
        }

        if (bezt) {
          /* Get mouse location in 3D space. */
          float location[3];
          copy_v3_v3(location, mouse_location_to_worldspace(event->mval, bezt->vec[1], vc));

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
          float *location = mouse_location_to_worldspace(event->mval, bp->vec, vc);

          copy_v3_v3(bp->vec, location);

          BKE_nurb_handles_calc(nu);
        }
      }
    }
  }
  else if (ELEM(event->type, LEFTMOUSE)) {
    if (event->val == KM_PRESS) {
      retval = ED_curve_editnurb_select_pick(C, mval, extend, deselect, toggle);
      RNA_boolean_set(op->ptr, "new", !retval);
      /* Check if point underneath mouse. Get point if any. */
      if (retval) {
        if (event->ctrl) {
          /* Delete retrieved control point. */
          short hand;

          copy_v2_v2_int(vc.mval, mval);
          ED_curve_pick_vert(&vc, 1, &nu, &bezt, &bp, &hand, &basact);
          if (bezt) {
            /* Copy all points to new BezTriple excluding selected point. */
            delete_bezt_from_nurb(bezt, nu);
          }
          else if (bp) {
            /* Copy all points to new BPoint excluding selected point. */
            delete_bp_from_nurb(bp, nu);
          }

          cu->actvert = CU_ACT_NONE;
          BKE_nurb_handles_calc(nu);
        }
      }
      else {
        if (event->ctrl) {
          /* If curve segment is nearby, add control point at the snapped point
          between the adjacent control points in the curve data structure. */
          EditNurb *editnurb = cu->editnurb;
          struct {
            int bezt_index;
            Nurb *nurb;
            float min_dist;
            bool has_prev, has_next;
            float prev_loc[3], cut_loc[3], next_loc[3];
            float mval[2];
          } data = {NULL};

          float threshold_dist = ED_view3d_select_dist_px();
          ListBase *nurbs = BKE_curve_editNurbs_get(cu);
          data.mval[0] = event->mval[0];
          data.mval[1] = event->mval[1];

          for (nu = nurbs->first; nu; nu = nu->next) {
            if (nu->type == CU_BEZIER) {
              float screen_co[2];
              if (data.nurb == NULL) {
                ED_view3d_project_float_object(vc.region,
                                               nu->bezt->vec[1],
                                               screen_co,
                                               V3D_PROJ_RET_CLIP_BB | V3D_PROJ_RET_CLIP_WIN) ==
                    V3D_PROJ_RET_OK;

                data.nurb = nu;
                data.bezt_index = 0;
                data.min_dist = len_manhattan_v2v2(screen_co, data.mval);
                copy_v3_v3(data.cut_loc, nu->bezt->vec);
              }
              for (int i = 0; i < nu->pntsu - 1; i++) {
                BezTriple *bezt = &nu->bezt[i];
                float resolu = nu->resolu;
                float *points = MEM_mallocN(sizeof(float[3]) * (resolu + 1), "makeCut_bezier");

                /* Calculate all points on curve. TODO: Get existing . */
                for (int j = 0; j < 3; j++) {
                  BKE_curve_forward_diff_bezier(bezt->vec[1][j],
                                                bezt->vec[2][j],
                                                (bezt + 1)->vec[0][j],
                                                (bezt + 1)->vec[1][j],
                                                points + j,
                                                resolu,
                                                sizeof(float[3]));
                }

                /* Calculate angle for middle points */
                for (int k = 0; k <= resolu; k++) {
                  /* Convert point to screen coordinates */
                  bool check = ED_view3d_project_float_object(vc.region,
                                                              points + 3 * k,
                                                              screen_co,
                                                              V3D_PROJ_RET_CLIP_BB |
                                                                  V3D_PROJ_RET_CLIP_WIN) ==
                               V3D_PROJ_RET_OK;
                  if (check) {
                    float distance = len_manhattan_v2v2(screen_co, data.mval);
                    if (distance < data.min_dist) {
                      data.min_dist = distance;
                      data.nurb = nu;
                      data.bezt_index = i;

                      copy_v3_v3(data.cut_loc, points + 3 * k);

                      data.has_prev = k > 0;
                      data.has_next = k < resolu;
                      if (data.has_prev) {
                        copy_v3_v3(data.prev_loc, points + 3 * (k - 1));
                      }
                      if (data.has_next) {
                        copy_v3_v3(data.next_loc, points + 3 * (k + 1));
                      }
                    }
                  }
                }
                MEM_freeN(points);
              }
            }
            else if (nu->type == CU_NURBS) {
            }
          }
          /* If the minimum distance found < threshold distance, make cut. */
          if (data.min_dist < 300) {
            int index = data.bezt_index;
            nu = data.nurb;
            float *cut_loc;
            if (nu->bezt) {
              if (data.has_prev) {
                float *point = get_closest_point_on_edge(
                    data.mval, data.cut_loc, data.prev_loc, vc);
                float point_2d[2];
                ED_view3d_project_float_object(
                    vc.region, point, point_2d, V3D_PROJ_RET_CLIP_BB | V3D_PROJ_RET_CLIP_WIN) ==
                    V3D_PROJ_RET_OK;
                float dist = len_manhattan_v2v2(point_2d, data.mval);
                if (dist < data.min_dist) {
                  data.min_dist = dist;
                  copy_v3_v3(data.cut_loc, point);
                }
              }
              if (data.has_next) {
                float *point = get_closest_point_on_edge(
                    data.mval, data.cut_loc, data.next_loc, vc);
                float point_2d[2];
                ED_view3d_project_float_object(
                    vc.region, point, point_2d, V3D_PROJ_RET_CLIP_BB | V3D_PROJ_RET_CLIP_WIN) ==
                    V3D_PROJ_RET_OK;
                float dist = len_manhattan_v2v2(point_2d, data.mval);
                if (dist < data.min_dist) {
                  data.min_dist = dist;
                  copy_v3_v3(data.cut_loc, point);
                }
              }

              index++;

              BezTriple *bezt1 = (BezTriple *)MEM_mallocN((nu->pntsu + 1) * sizeof(BezTriple),
                                                          "delNurb");
              /* Copy all control points before the cut to the new memory. */
              memcpy(bezt1, nu->bezt, index * sizeof(BezTriple));
              BezTriple *new_bezt = bezt1 + index;

              /* Duplicate control point after the cut. */
              // ED_curve_beztcpy(editnurb, new_bezt, new_bezt - 1, 1);
              memcpy(new_bezt, new_bezt - 1, sizeof(BezTriple));
              new_bezt->h1 = new_bezt->h2 = HD_AUTO;
              copy_v3_v3(new_bezt->vec[1], data.cut_loc);

              /* Copy all control points after the cut to the new memory. */
              memcpy(bezt1 + index + 1, nu->bezt + index, (nu->pntsu - index) * sizeof(BezTriple));
              nu->pntsu += 1;
              cu->actvert = CU_ACT_NONE;

              MEM_freeN(nu->bezt);
              nu->bezt = bezt1;
              ED_curve_deselect_all(editnurb);
              BKE_nurb_handles_calc(nu);
              new_bezt->f1 = new_bezt->f2 = new_bezt->f3 = 1;
            }
            else if (nu->bp) {
              BPoint *bp1 = (BPoint *)MEM_mallocN((nu->pntsu + 1) * sizeof(BPoint), "delNurb");
              memcpy(bp1, nu->bp, index * sizeof(BPoint));
              BPoint *new_bp = bp1 + index;

              // ED_curve_bpcpy(editnurb, new_bp, new_bp - 1, 1);
              memcpy(new_bp, new_bp - 1, sizeof(BPoint));
              copy_v3_v3(new_bp->vec, data.cut_loc);

              memcpy(bp1 + index + 1, nu->bp + index, (nu->pntsu - index) * sizeof(BPoint));
              nu->pntsu += 1;
              cu->actvert = CU_ACT_NONE;

              MEM_freeN(nu->bp);
              nu->bp = bp1;
              ED_curve_deselect_all(editnurb);
              BKE_nurb_handles_calc(nu);
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
          ed_editcurve_addvert(C, location);
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

static int curve_pen_exec(bContext *C, wmOperator *op)
{
}

static int curve_pen_invoke(bContext *C, wmOperator *op, const wmEvent *event)
{
  RNA_int_set_array(op->ptr, "location", event->mval);

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
  ot->exec = curve_pen_exec;
  ot->invoke = curve_pen_invoke;
  ot->modal = curve_pen_modal;
  ot->poll = ED_operator_view3d_active;

  /* flags */
  ot->flag = OPTYPE_UNDO;

  /* properties */
  WM_operator_properties_mouse_select(ot);

  PropertyRNA *prop;
  prop = RNA_def_boolean(
      ot->srna,
      "center",
      0,
      "Center",
      "Use the object center when selecting, in edit mode used to extend object selection");
  RNA_def_property_flag(prop, PROP_SKIP_SAVE);
  prop = RNA_def_boolean(ot->srna, "object", 0, "Object", "Use object selection (edit mode only)");
  RNA_def_property_flag(prop, PROP_SKIP_SAVE);
  prop = RNA_def_boolean(ot->srna, "dragging", 0, "Dragging", "Check if click and drag");
  RNA_def_property_flag(prop, PROP_SKIP_SAVE);
  prop = RNA_def_boolean(
      ot->srna, "new", 0, "New Point Drag", "The point was added with the press before drag");
  RNA_def_property_flag(prop, PROP_SKIP_SAVE);

  prop = RNA_def_int_vector(ot->srna,
                            "location",
                            2,
                            NULL,
                            INT_MIN,
                            INT_MAX,
                            "Location",
                            "Mouse location",
                            INT_MIN,
                            INT_MAX);
  RNA_def_property_flag(prop, PROP_HIDDEN);

  prop = RNA_def_boolean(ot->srna, "wait_for_input", true, "Wait for Input", "");
  RNA_def_property_flag(prop, PROP_HIDDEN | PROP_SKIP_SAVE);
}
