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
 * \ingroup wm
 *
 * \name Window-Manager XR Operators
 *
 * Collection of XR-related operators.
 */

#include "BLI_kdopbvh.h"
#include "BLI_listbase.h"
#include "BLI_math.h"

#include "BKE_context.h"
#include "BKE_editmesh.h"
#include "BKE_global.h"
#include "BKE_layer.h"
#include "BKE_main.h"
#include "BKE_object.h"
#include "BKE_screen.h"

#include "DEG_depsgraph.h"
#include "DEG_depsgraph_query.h"

#include "ED_keyframing.h"
#include "ED_mesh.h"
#include "ED_object.h"
#include "ED_screen.h"
#include "ED_select_utils.h"
#include "ED_space_api.h"
#include "ED_transform_snap_object_context.h"
#include "ED_view3d.h"

#include "GHOST_Types.h"

#include "GPU_immediate.h"

#include "MEM_guardedalloc.h"

#include "RNA_access.h"
#include "RNA_define.h"

#include "WM_api.h"
#include "WM_types.h"

#include "wm_xr_intern.h"

/* -------------------------------------------------------------------- */
/** \name Operator Callbacks
 * \{ */

/* op->poll */
static bool wm_xr_operator_sessionactive(bContext *C)
{
  wmWindowManager *wm = CTX_wm_manager(C);
  if (WM_xr_session_is_ready(&wm->xr)) {
    return true;
  }
  return false;
}

/* -------------------------------------------------------------------- */
/** \name XR Session Toggle
 *
 * Toggles an XR session, creating an XR context if necessary.
 * \{ */

static void wm_xr_session_update_screen(Main *bmain, const wmXrData *xr_data)
{
  const bool session_exists = WM_xr_session_exists(xr_data);

  for (bScreen *screen = bmain->screens.first; screen; screen = screen->id.next) {
    LISTBASE_FOREACH (ScrArea *, area, &screen->areabase) {
      LISTBASE_FOREACH (SpaceLink *, slink, &area->spacedata) {
        if (slink->spacetype == SPACE_VIEW3D) {
          View3D *v3d = (View3D *)slink;

          if (v3d->flag & V3D_XR_SESSION_MIRROR) {
            ED_view3d_xr_mirror_update(area, v3d, session_exists);
          }

          if (session_exists) {
            wmWindowManager *wm = bmain->wm.first;
            const Scene *scene = WM_windows_scene_get_from_screen(wm, screen);

            ED_view3d_xr_shading_update(wm, v3d, scene);
          }
          /* Ensure no 3D View is tagged as session root. */
          else {
            v3d->runtime.flag &= ~V3D_RUNTIME_XR_SESSION_ROOT;
          }
        }
      }
    }
  }

  WM_main_add_notifier(NC_WM | ND_XR_DATA_CHANGED, NULL);
}

static void wm_xr_session_update_screen_on_exit_cb(const wmXrData *xr_data)
{
  /* Just use G_MAIN here, storing main isn't reliable enough on file read or exit. */
  wm_xr_session_update_screen(G_MAIN, xr_data);
}

static int wm_xr_session_toggle_exec(bContext *C, wmOperator *UNUSED(op))
{
  Main *bmain = CTX_data_main(C);
  wmWindowManager *wm = CTX_wm_manager(C);
  wmWindow *win = CTX_wm_window(C);
  View3D *v3d = CTX_wm_view3d(C);

  /* Lazy-create xr context - tries to dynlink to the runtime, reading active_runtime.json. */
  if (wm_xr_init(wm) == false) {
    return OPERATOR_CANCELLED;
  }

  v3d->runtime.flag |= V3D_RUNTIME_XR_SESSION_ROOT;
  wm_xr_session_toggle(C, wm, win, wm_xr_session_update_screen_on_exit_cb);
  wm_xr_session_update_screen(bmain, &wm->xr);

  WM_event_add_notifier(C, NC_WM | ND_XR_DATA_CHANGED, NULL);

  return OPERATOR_FINISHED;
}

static void WM_OT_xr_session_toggle(wmOperatorType *ot)
{
  /* identifiers */
  ot->name = "Toggle VR Session";
  ot->idname = "WM_OT_xr_session_toggle";
  ot->description =
      "Open a view for use with virtual reality headsets, or close it if already "
      "opened";

  /* callbacks */
  ot->exec = wm_xr_session_toggle_exec;
  ot->poll = ED_operator_view3d_active;

  /* XXX INTERNAL just to hide it from the search menu by default, an Add-on will expose it in the
   * UI instead. Not meant as a permanent solution. */
  ot->flag = OPTYPE_INTERNAL;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name XR Raycast Select
 *
 * Casts a ray from an XR controller's pose and selects any hit geometry.
 * \{ */

typedef struct XrRaycastSelectData {
  float origin[3];
  float direction[3];
  float end[3];
  void *draw_handle;
} XrRaycastSelectData;

static void wm_xr_select_raycast_draw(const bContext *UNUSED(C),
                                      ARegion *UNUSED(region),
                                      void *customdata)
{
  const XrRaycastSelectData *data = customdata;

  GPUVertFormat *format = immVertexFormat();
  uint pos = GPU_vertformat_attr_add(format, "pos", GPU_COMP_F32, 3, GPU_FETCH_FLOAT);
  immBindBuiltinProgram(GPU_SHADER_3D_UNIFORM_COLOR);
  immUniformColor4f(0.863f, 0.0f, 0.545f, 1.0f);

  GPU_depth_test(GPU_DEPTH_LESS_EQUAL);
  GPU_line_width(3.0f);

  immBegin(GPU_PRIM_LINES, 2);
  immVertex3fv(pos, data->origin);
  immVertex3fv(pos, data->end);
  immEnd();

  immUnbindProgram();
}

static void wm_xr_select_raycast_init(wmOperator *op)
{
  BLI_assert(op->customdata == NULL);

  op->customdata = MEM_callocN(sizeof(XrRaycastSelectData), __func__);

  SpaceType *st = BKE_spacetype_from_id(SPACE_VIEW3D);
  if (st) {
    ARegionType *art = BKE_regiontype_from_id(st, RGN_TYPE_XR);
    if (art) {
      ((XrRaycastSelectData *)op->customdata)->draw_handle = ED_region_draw_cb_activate(
          art, wm_xr_select_raycast_draw, op->customdata, REGION_DRAW_POST_VIEW);
    }
  }
}

static void wm_xr_select_raycast_uninit(wmOperator *op)
{
  if (op->customdata) {
    SpaceType *st = BKE_spacetype_from_id(SPACE_VIEW3D);
    if (st) {
      ARegionType *art = BKE_regiontype_from_id(st, RGN_TYPE_XR);
      if (art) {
        ED_region_draw_cb_exit(art, ((XrRaycastSelectData *)op->customdata)->draw_handle);
      }
    }

    MEM_freeN(op->customdata);
  }
}

typedef enum eXrSelectElem {
  XR_SEL_BASE = 0,
  XR_SEL_VERTEX = 1,
  XR_SEL_EDGE = 2,
  XR_SEL_FACE = 3,
} eXrSelectElem;

static void wm_xr_select_op_apply(void *elem,
                                  BMesh *bm,
                                  eXrSelectElem select_elem,
                                  eSelectOp select_op,
                                  bool *r_changed,
                                  bool *r_set)
{
  const bool selected_prev = (select_elem == XR_SEL_BASE) ?
                                 (((Base *)elem)->flag & BASE_SELECTED) != 0 :
                                 (((BMElem *)elem)->head.hflag & BM_ELEM_SELECT) != 0;

  if (selected_prev) {
    switch (select_op) {
      case SEL_OP_SUB:
      case SEL_OP_XOR: {
        switch (select_elem) {
          case XR_SEL_BASE:
            ED_object_base_select((Base *)elem, BA_DESELECT);
            *r_changed = true;
            break;
          case XR_SEL_VERTEX:
            BM_vert_select_set(bm, (BMVert *)elem, false);
            *r_changed = true;
            break;
          case XR_SEL_EDGE:
            BM_edge_select_set(bm, (BMEdge *)elem, false);
            *r_changed = true;
            break;
          case XR_SEL_FACE:
            BM_face_select_set(bm, (BMFace *)elem, false);
            *r_changed = true;
            break;
        }
        break;
      }
      default: {
        break;
      }
    }
  }
  else {
    switch (select_op) {
      case SEL_OP_SET:
      case SEL_OP_ADD:
      case SEL_OP_XOR: {
        switch (select_elem) {
          case XR_SEL_BASE:
            ED_object_base_select((Base *)elem, BA_SELECT);
            *r_changed = true;
            break;
          case XR_SEL_VERTEX:
            BM_vert_select_set(bm, (BMVert *)elem, true);
            *r_changed = true;
            break;
          case XR_SEL_EDGE:
            BM_edge_select_set(bm, (BMEdge *)elem, true);
            *r_changed = true;
            break;
          case XR_SEL_FACE:
            BM_face_select_set(bm, (BMFace *)elem, true);
            *r_changed = true;
            break;
        }
      }
      default: {
        break;
      }
    }

    if (select_op == SEL_OP_SET) {
      *r_set = true;
    }
  }
}

static bool wm_xr_select_raycast(bContext *C,
                                 const float origin[3],
                                 const float direction[3],
                                 float *ray_dist,
                                 eSelectOp select_op,
                                 bool deselect_all,
                                 bool selectable_only)
{
  /* Uses same raycast method as Scene.ray_cast(). */
  Depsgraph *depsgraph = CTX_data_ensure_evaluated_depsgraph(C);
  ViewContext vc;
  ED_view3d_viewcontext_init(C, &vc, depsgraph);
  vc.em = (vc.obedit && (vc.obedit->type == OB_MESH)) ? BKE_editmesh_from_object(vc.obedit) : NULL;

  float location[3];
  float normal[3];
  int index;
  Object *ob = NULL;
  float obmat[4][4];

  SnapObjectContext *sctx = ED_transform_snap_object_context_create(vc.scene, 0);

  ED_transform_snap_object_project_ray_ex(
      sctx,
      depsgraph,
      &(const struct SnapObjectParams){
          .snap_select = vc.em ? SNAP_SELECTED : (selectable_only ? SNAP_SELECTABLE : SNAP_ALL)},
      origin,
      direction,
      ray_dist,
      location,
      normal,
      &index,
      &ob,
      obmat);

  ED_transform_snap_object_context_destroy(sctx);

  /* Select. */
  bool hit = false;
  bool changed = false;

  if (ob && vc.em &&
      ((ob == vc.obedit) || (ob->id.orig_id == &vc.obedit->id))) { /* TODO_XR: Non-mesh objects. */
    BMesh *bm = vc.em->bm;
    BMFace *f = NULL;
    BMEdge *e = NULL;
    BMVert *v = NULL;

    if (index != -1) {
      ToolSettings *ts = vc.scene->toolsettings;
      float co[3];
      f = BM_face_at_index(bm, index);

      if ((ts->selectmode & SCE_SELECT_VERTEX) != 0) {
        /* Find nearest vertex. */
        float dist_max = *ray_dist;
        float dist;
        BMLoop *l = f->l_first;
        for (int i = 0; i < f->len; ++i, l = l->next) {
          mul_v3_m4v3(co, obmat, l->v->co);
          if ((dist = len_manhattan_v3v3(location, co)) < dist_max) {
            v = l->v;
            dist_max = dist;
          }
        }
        if (v) {
          hit = true;
        }
      }
      if ((ts->selectmode & SCE_SELECT_EDGE) != 0) {
        /* Find nearest edge. */
        float dist_max = *ray_dist;
        float dist;
        BMLoop *l = f->l_first;
        for (int i = 0; i < f->len; ++i, l = l->next) {
          add_v3_v3v3(co, l->e->v1->co, l->e->v2->co);
          mul_v3_fl(co, 0.5f);
          mul_m4_v3(obmat, co);
          if ((dist = len_manhattan_v3v3(location, co)) < dist_max) {
            e = l->e;
            dist_max = dist;
          }
        }
        if (e) {
          hit = true;
        }
      }
      if ((ts->selectmode & SCE_SELECT_FACE) != 0) {
        hit = true;
      }
      else {
        f = NULL;
      }
    }

    if (!hit) {
      if (deselect_all) {
        changed = EDBM_mesh_deselect_all_multi(C);
      }
    }
    else {
      bool set_v = false;
      bool set_e = false;
      bool set_f = false;

      if (v) {
        wm_xr_select_op_apply(v, bm, XR_SEL_VERTEX, select_op, &changed, &set_v);
      }
      if (e) {
        wm_xr_select_op_apply(e, bm, XR_SEL_EDGE, select_op, &changed, &set_e);
      }
      if (f) {
        wm_xr_select_op_apply(f, bm, XR_SEL_FACE, select_op, &changed, &set_f);
      }

      if (set_v || set_e || set_f) {
        EDBM_mesh_deselect_all_multi(C);
        if (set_v) {
          BM_vert_select_set(bm, v, true);
        }
        if (set_e) {
          BM_edge_select_set(bm, e, true);
        }
        if (set_f) {
          BM_face_select_set(bm, f, true);
        }
      }
    }

    if (changed) {
      DEG_id_tag_update((ID *)vc.obedit->data, ID_RECALC_SELECT);
      WM_event_add_notifier(C, NC_GEOM | ND_SELECT, vc.obedit->data);
    }
  }
  else if (vc.em) {
    if (deselect_all) {
      changed = EDBM_mesh_deselect_all_multi(C);
    }

    if (changed) {
      DEG_id_tag_update((ID *)vc.obedit->data, ID_RECALC_SELECT);
      WM_event_add_notifier(C, NC_GEOM | ND_SELECT, vc.obedit->data);
    }
  }
  else {
    if (ob) {
      hit = true;
    }

    if (!hit) {
      if (deselect_all) {
        changed = object_deselect_all_except(vc.view_layer, NULL);
      }
    }
    else {
      Base *base = BKE_view_layer_base_find(vc.view_layer, DEG_get_original_object(ob));
      if (base && BASE_SELECTABLE(vc.v3d, base)) {
        bool set = false;
        wm_xr_select_op_apply(base, NULL, XR_SEL_BASE, select_op, &changed, &set);
        if (set) {
          object_deselect_all_except(vc.view_layer, base);
        }
      }
    }

    if (changed) {
      DEG_id_tag_update(&vc.scene->id, ID_RECALC_SELECT);
      WM_event_add_notifier(C, NC_SCENE | ND_OB_SELECT, vc.scene);
    }
  }

  return changed;
}

static int wm_xr_select_raycast_invoke_3d(bContext *C, wmOperator *op, const wmEvent *event)
{
  BLI_assert(event->type == EVT_XR_ACTION);
  BLI_assert(event->custom == EVT_DATA_XR);
  BLI_assert(event->customdata);

  wm_xr_select_raycast_init(op);

  int retval = op->type->modal_3d(C, op, event);

  if ((retval & OPERATOR_RUNNING_MODAL) != 0) {
    WM_event_add_modal_handler(C, op);
  }

  return retval;
}

static int wm_xr_select_raycast_exec(bContext *UNUSED(C), wmOperator *UNUSED(op))
{
  return OPERATOR_CANCELLED;
}

static int wm_xr_select_raycast_modal_3d(bContext *C, wmOperator *op, const wmEvent *event)
{
  BLI_assert(event->type == EVT_XR_ACTION);
  BLI_assert(event->custom == EVT_DATA_XR);
  BLI_assert(event->customdata);

  wmWindowManager *wm = CTX_wm_manager(C);
  wmXrActionData *actiondata = event->customdata;
  XrRaycastSelectData *data = op->customdata;
  float axis[3];

  PropertyRNA *prop = RNA_struct_find_property(op->ptr, "axis");
  if (prop) {
    RNA_property_float_get_array(op->ptr, prop, axis);
    normalize_v3(axis);
  }
  else {
    axis[0] = 0.0f;
    axis[1] = 0.0f;
    axis[2] = -1.0f;
  }

  copy_v3_v3(data->origin, actiondata->controller_loc);

  mul_qt_v3(actiondata->controller_rot, axis);
  copy_v3_v3(data->direction, axis);

  mul_v3_v3fl(data->end, data->direction, wm->xr.session_settings.clip_end);
  add_v3_v3(data->end, data->origin);

  if (event->val == KM_PRESS) {
    return OPERATOR_RUNNING_MODAL;
  }
  else if (event->val == KM_RELEASE) {
    float ray_dist;
    eSelectOp select_op = SEL_OP_SET;
    bool deselect_all, selectable_only;
    bool ret;

    prop = RNA_struct_find_property(op->ptr, "distance");
    ray_dist = prop ? RNA_property_float_get(op->ptr, prop) : BVH_RAYCAST_DIST_MAX;

    prop = RNA_struct_find_property(op->ptr, "toggle");
    if (prop && RNA_property_boolean_get(op->ptr, prop)) {
      select_op = SEL_OP_XOR;
    }
    prop = RNA_struct_find_property(op->ptr, "deselect");
    if (prop && RNA_property_boolean_get(op->ptr, prop)) {
      select_op = SEL_OP_SUB;
    }
    prop = RNA_struct_find_property(op->ptr, "extend");
    if (prop && RNA_property_boolean_get(op->ptr, prop)) {
      select_op = SEL_OP_ADD;
    }

    prop = RNA_struct_find_property(op->ptr, "deselect_all");
    deselect_all = prop ? RNA_property_boolean_get(op->ptr, prop) : false;

    prop = RNA_struct_find_property(op->ptr, "selectable_only");
    selectable_only = prop ? RNA_property_boolean_get(op->ptr, prop) : true;

    ret = wm_xr_select_raycast(
        C, data->origin, data->direction, &ray_dist, select_op, deselect_all, selectable_only);

    wm_xr_select_raycast_uninit(op);

    return ret ? OPERATOR_FINISHED : OPERATOR_CANCELLED;
  }

  /* XR events currently only support press and release. */
  BLI_assert(false);
  wm_xr_select_raycast_uninit(op);
  return OPERATOR_CANCELLED;
}

static void WM_OT_xr_select_raycast(wmOperatorType *ot)
{
  /* identifiers */
  ot->name = "XR Raycast Select";
  ot->idname = "WM_OT_xr_select_raycast";
  ot->description = "Raycast select with a VR controller";

  /* callbacks */
  ot->invoke_3d = wm_xr_select_raycast_invoke_3d;
  ot->exec = wm_xr_select_raycast_exec;
  ot->modal_3d = wm_xr_select_raycast_modal_3d;
  ot->poll = wm_xr_operator_sessionactive;

  /* flags */
  ot->flag = OPTYPE_UNDO;

  /* properties */
  static const float default_axis[3] = {0.0f, 0.0f, -1.0f};

  WM_operator_properties_mouse_select(ot);

  /* Override "deselect_all" default value. */
  PropertyRNA *prop = RNA_struct_type_find_property(ot->srna, "deselect_all");
  BLI_assert(prop != NULL);
  RNA_def_property_boolean_default(prop, true);

  RNA_def_float(ot->srna,
                "distance",
                BVH_RAYCAST_DIST_MAX,
                0.0,
                BVH_RAYCAST_DIST_MAX,
                "",
                "Maximum distance",
                0.0,
                BVH_RAYCAST_DIST_MAX);
  RNA_def_float_vector(ot->srna,
                       "axis",
                       3,
                       default_axis,
                       -1.0f,
                       1.0f,
                       "Axis",
                       "Raycast axis in controller space",
                       -1.0f,
                       1.0f);
  RNA_def_boolean(ot->srna,
                  "selectable_only",
                  true,
                  "Selectable Only",
                  "Only allow selectable objects to influence raycast result");
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name XR Grab
 *
 * Transforms location and rotation of selected objects relative to an XR controller's pose.
 * \{ */

typedef struct XrGrabData {
  float mat_prev[4][4];
} XrGrabData;

static void wm_xr_grab_init(wmOperator *op)
{
  BLI_assert(op->customdata == NULL);

  op->customdata = MEM_callocN(sizeof(XrGrabData), __func__);
}

static void wm_xr_grab_uninit(wmOperator *op)
{
  if (op->customdata) {
    MEM_freeN(op->customdata);
  }
}

static int wm_xr_grab_invoke_3d(bContext *C, wmOperator *op, const wmEvent *event)
{
  BLI_assert(event->type == EVT_XR_ACTION);
  BLI_assert(event->custom == EVT_DATA_XR);
  BLI_assert(event->customdata);

  bool loc_lock, rot_lock;
  float loc_t, rot_t;
  float loc_ofs[3], rot_ofs[4];
  bool loc_ofs_set = false;
  bool rot_ofs_set = false;

  PropertyRNA *prop = RNA_struct_find_property(op->ptr, "location_lock");
  loc_lock = prop ? RNA_property_boolean_get(op->ptr, prop) : false;
  if (!loc_lock) {
    prop = RNA_struct_find_property(op->ptr, "location_interpolation");
    loc_t = prop ? RNA_property_float_get(op->ptr, prop) : 0.0f;
    prop = RNA_struct_find_property(op->ptr, "location_offset");
    if (prop && RNA_property_is_set(op->ptr, prop)) {
      RNA_property_float_get_array(op->ptr, prop, loc_ofs);
      loc_ofs_set = true;
    }
  }

  prop = RNA_struct_find_property(op->ptr, "rotation_lock");
  rot_lock = prop ? RNA_property_boolean_get(op->ptr, prop) : false;
  if (!rot_lock) {
    prop = RNA_struct_find_property(op->ptr, "rotation_interpolation");
    rot_t = prop ? RNA_property_float_get(op->ptr, prop) : 0.0f;
    prop = RNA_struct_find_property(op->ptr, "rotation_offset");
    if (prop && RNA_property_is_set(op->ptr, prop)) {
      float tmp[3];
      RNA_property_float_get_array(op->ptr, prop, tmp);
      eul_to_quat(rot_ofs, tmp);
      normalize_qt(rot_ofs);
      rot_ofs_set = true;
    }
  }

  if (loc_lock && rot_lock) {
    return OPERATOR_CANCELLED;
  }

  const wmXrActionData *actiondata = event->customdata;
  Object *obedit = CTX_data_edit_object(C);
  BMEditMesh *em = (obedit && (obedit->type == OB_MESH)) ? BKE_editmesh_from_object(obedit) : NULL;
  float tmp0[4], tmp1[4], tmp2[4];
  bool selected = false;

  if (loc_ofs_set) {
    /* Convert to controller space. */
    float tmp[3][3];
    quat_to_mat3(tmp, actiondata->controller_rot);
    mul_m3_v3(tmp, loc_ofs);
  }
  if (rot_ofs_set) {
    /* Convert to controller space. */
    invert_qt_qt_normalized(tmp0, rot_ofs);
    mul_qt_qtqt(rot_ofs, actiondata->controller_rot, tmp0);
    normalize_qt(rot_ofs);
  }

  if (em) { /* TODO_XR: Non-mesh objects. */
    /* Check for selection. */
    Scene *scene = CTX_data_scene(C);
    ToolSettings *ts = scene->toolsettings;
    BMesh *bm = em->bm;
    BMIter iter;
    if ((ts->selectmode & SCE_SELECT_FACE) != 0) {
      BMFace *f;
      BM_ITER_MESH (f, &iter, bm, BM_FACES_OF_MESH) {
        if (BM_elem_flag_test(f, BM_ELEM_SELECT)) {
          selected = true;
          break;
        }
      }
    }
    if (!selected) {
      if ((ts->selectmode & SCE_SELECT_EDGE) != 0) {
        BMEdge *e;
        BM_ITER_MESH (e, &iter, bm, BM_EDGES_OF_MESH) {
          if (BM_elem_flag_test(e, BM_ELEM_SELECT)) {
            selected = true;
            break;
          }
        }
      }
      if (!selected) {
        if ((ts->selectmode & SCE_SELECT_VERTEX) != 0) {
          BMVert *v;
          BM_ITER_MESH (v, &iter, bm, BM_VERTS_OF_MESH) {
            if (BM_elem_flag_test(v, BM_ELEM_SELECT)) {
              selected = true;
              break;
            }
          }
        }
      }
    }
  }
  else {
    /* Apply interpolation and offsets. */
    CTX_DATA_BEGIN (C, Object *, ob, selected_objects) {
      bool update = false;

      if (!loc_lock) {
        if (loc_t > 0.0f) {
          ob->loc[0] += loc_t * (actiondata->controller_loc[0] - ob->loc[0]);
          ob->loc[1] += loc_t * (actiondata->controller_loc[1] - ob->loc[1]);
          ob->loc[2] += loc_t * (actiondata->controller_loc[2] - ob->loc[2]);
          update = true;
        }
        if (loc_ofs_set) {
          add_v3_v3(ob->loc, loc_ofs);
          update = true;
        }
      }

      if (!rot_lock) {
        if (rot_t > 0.0f) {
          eul_to_quat(tmp1, ob->rot);
          interp_qt_qtqt(tmp0, tmp1, actiondata->controller_rot, rot_t);
          if (!rot_ofs_set) {
            quat_to_eul(ob->rot, tmp0);
          }
          update = true;
        }
        else if (rot_ofs_set) {
          eul_to_quat(tmp0, ob->rot);
        }
        if (rot_ofs_set) {
          rotation_between_quats_to_quat(tmp1, rot_ofs, tmp0);
          mul_qt_qtqt(tmp0, rot_ofs, tmp1);
          normalize_qt(tmp0);
          mul_qt_qtqt(tmp2, actiondata->controller_rot, tmp1);
          normalize_qt(tmp2);
          rotation_between_quats_to_quat(tmp1, tmp0, tmp2);

          mul_qt_qtqt(tmp2, tmp0, tmp1);
          normalize_qt(tmp2);
          quat_to_eul(ob->rot, tmp2);
          update = true;
        }
      }

      if (update) {
        DEG_id_tag_update(&ob->id, ID_RECALC_TRANSFORM);
      }
      selected = true;
    }
    CTX_DATA_END;
  }

  if (!selected) {
    wm_xr_grab_uninit(op);
    return OPERATOR_CANCELLED;
  }

  wm_xr_grab_init(op);

  XrGrabData *data = op->customdata;
  quat_to_mat4(data->mat_prev, actiondata->controller_rot);
  copy_v3_v3(data->mat_prev[3], actiondata->controller_loc);

  WM_event_add_modal_handler(C, op);

  return OPERATOR_RUNNING_MODAL;
}

static int wm_xr_grab_exec(bContext *UNUSED(C), wmOperator *UNUSED(op))
{
  return OPERATOR_CANCELLED;
}

static int wm_xr_grab_modal_3d(bContext *C, wmOperator *op, const wmEvent *event)
{
  BLI_assert(event->type == EVT_XR_ACTION);
  BLI_assert(event->custom == EVT_DATA_XR);
  BLI_assert(event->customdata);

  const wmXrActionData *actiondata = event->customdata;
  XrGrabData *data = op->customdata;
  Scene *scene = CTX_data_scene(C);
  ViewLayer *view_layer = CTX_data_view_layer(C);
  Object *obedit = CTX_data_edit_object(C);
  BMEditMesh *em = (obedit && (obedit->type == OB_MESH)) ? BKE_editmesh_from_object(obedit) : NULL;
  wmWindowManager *wm = CTX_wm_manager(C);
  bScreen *screen_anim = ED_screen_animation_playing(wm);
  bool loc_lock, rot_lock;
  bool selected = false;
  float delta[4][4], tmp0[4][4], tmp1[4][4], tmp2[4][4];

  PropertyRNA *prop = RNA_struct_find_property(op->ptr, "location_lock");
  loc_lock = prop ? RNA_property_boolean_get(op->ptr, prop) : false;
  prop = RNA_struct_find_property(op->ptr, "rotation_lock");
  rot_lock = prop ? RNA_property_boolean_get(op->ptr, prop) : false;

  if (em) { /* TODO_XR: Non-mesh objects. */
    if (!loc_lock || !rot_lock) {
      ToolSettings *ts = scene->toolsettings;
      BMesh *bm = em->bm;
      BMIter iter;

      if (rot_lock) {
        unit_m4(tmp0);
        copy_v3_v3(tmp0[3], data->mat_prev[3]);
        mul_m4_m4m4(tmp1, obedit->imat, tmp0);
        invert_m4(tmp1);

        quat_to_mat4(data->mat_prev, actiondata->controller_rot);
        copy_v3_v3(data->mat_prev[3], actiondata->controller_loc);
        copy_v3_v3(tmp0[3], data->mat_prev[3]);
        mul_m4_m4m4(tmp2, obedit->imat, tmp0);

        mul_m4_m4m4(delta, tmp2, tmp1);
      }
      else {
        copy_m4_m4(tmp0, data->mat_prev);
        mul_m4_m4m4(tmp1, obedit->imat, tmp0);
        invert_m4(tmp1);

        quat_to_mat4(data->mat_prev, actiondata->controller_rot);
        copy_v3_v3(data->mat_prev[3], actiondata->controller_loc);
        copy_m4_m4(tmp0, data->mat_prev);
        mul_m4_m4m4(tmp2, obedit->imat, tmp0);

        mul_m4_m4m4(delta, tmp2, tmp1);

        if (loc_lock) {
          zero_v3(delta[3]);
        }
      }

      if ((ts->selectmode & SCE_SELECT_VERTEX) != 0) {
        BMVert *v;
        BM_ITER_MESH (v, &iter, bm, BM_VERTS_OF_MESH) {
          if (BM_elem_flag_test(v, BM_ELEM_SELECT) &&
              !BM_elem_flag_test(v, BM_ELEM_INTERNAL_TAG)) {
            mul_m4_v3(delta, v->co);
            BM_elem_flag_enable(v, BM_ELEM_INTERNAL_TAG);
          }
        }
      }
      if ((ts->selectmode & SCE_SELECT_EDGE) != 0) {
        BMEdge *e;
        BM_ITER_MESH (e, &iter, bm, BM_EDGES_OF_MESH) {
          if (BM_elem_flag_test(e, BM_ELEM_SELECT)) {
            if (!BM_elem_flag_test(e->v1, BM_ELEM_INTERNAL_TAG)) {
              mul_m4_v3(delta, e->v1->co);
              BM_elem_flag_enable(e->v1, BM_ELEM_INTERNAL_TAG);
            }
            if (!BM_elem_flag_test(e->v2, BM_ELEM_INTERNAL_TAG)) {
              mul_m4_v3(delta, e->v2->co);
              BM_elem_flag_enable(e->v2, BM_ELEM_INTERNAL_TAG);
            }
          }
        }
      }
      if ((ts->selectmode & SCE_SELECT_FACE) != 0) {
        BMFace *f;
        BMLoop *l;
        BM_ITER_MESH (f, &iter, bm, BM_FACES_OF_MESH) {
          if (BM_elem_flag_test(f, BM_ELEM_SELECT)) {
            l = f->l_first;
            for (int i = 0; i < f->len; ++i, l = l->next) {
              if (!BM_elem_flag_test(l->v, BM_ELEM_INTERNAL_TAG)) {
                mul_m4_v3(delta, l->v->co);
                BM_elem_flag_enable(l->v, BM_ELEM_INTERNAL_TAG);
              }
            }
          }
        }
      }

      BM_mesh_elem_hflag_disable_all(bm, BM_VERT, BM_ELEM_INTERNAL_TAG, false);
      EDBM_mesh_normals_update(em);
      DEG_id_tag_update(&obedit->id, ID_RECALC_GEOMETRY);
    }
    else {
      quat_to_mat4(data->mat_prev, actiondata->controller_rot);
      copy_v3_v3(data->mat_prev[3], actiondata->controller_loc);
    }

    selected = true;
  }
  else {
    if (!loc_lock || !rot_lock) {
      if (rot_lock) {
        unit_m4(tmp0);
        copy_v3_v3(tmp0[3], data->mat_prev[3]);
        invert_m4_m4(tmp1, tmp0);

        quat_to_mat4(data->mat_prev, actiondata->controller_rot);
        copy_v3_v3(data->mat_prev[3], actiondata->controller_loc);
        copy_v3_v3(tmp0[3], data->mat_prev[3]);

        mul_m4_m4m4(delta, tmp0, tmp1);
      }
      else {
        invert_m4_m4(tmp0, data->mat_prev);
        quat_to_mat4(data->mat_prev, actiondata->controller_rot);
        copy_v3_v3(data->mat_prev[3], actiondata->controller_loc);
        mul_m4_m4m4(delta, data->mat_prev, tmp0);

        if (loc_lock) {
          zero_v3(delta[3]);
        }
      }
    }
    else {
      quat_to_mat4(data->mat_prev, actiondata->controller_rot);
      copy_v3_v3(data->mat_prev[3], actiondata->controller_loc);
    }

    CTX_DATA_BEGIN (C, Object *, ob, selected_objects) {
      if (!loc_lock || !rot_lock) {
        mul_m4_m4m4(tmp0, delta, ob->obmat);

        if (!loc_lock) {
          copy_v3_v3(ob->loc, tmp0[3]);
        }
        if (!rot_lock) {
          mat4_to_eul(ob->rot, tmp0);
        }

        DEG_id_tag_update(&ob->id, ID_RECALC_TRANSFORM);
      }

      if (screen_anim && autokeyframe_cfra_can_key(scene, &ob->id)) {
        wm_xr_session_object_autokey(C, scene, view_layer, NULL, ob, true);
      }

      selected = true;
    }
    CTX_DATA_END;
  }

  if (!selected || (event->val == KM_RELEASE)) {
    wm_xr_grab_uninit(op);

    if (obedit && em) {
      WM_event_add_notifier(C, NC_GEOM | ND_DATA, obedit->data);
    }
    else {
      WM_event_add_notifier(C, NC_SCENE | ND_TRANSFORM_DONE, scene);
    }
    return OPERATOR_FINISHED;
  }
  else if (event->val == KM_PRESS) {
    return OPERATOR_RUNNING_MODAL;
  }

  /* XR events currently only support press and release. */
  BLI_assert(false);
  wm_xr_grab_uninit(op);
  return OPERATOR_CANCELLED;
}

static void WM_OT_xr_grab(wmOperatorType *ot)
{
  /* identifiers */
  ot->name = "XR Grab";
  ot->idname = "WM_OT_xr_grab";
  ot->description =
      "Transform location and rotation of selected objects relative to a VR controller's pose";

  /* callbacks */
  ot->invoke_3d = wm_xr_grab_invoke_3d;
  ot->exec = wm_xr_grab_exec;
  ot->modal_3d = wm_xr_grab_modal_3d;
  ot->poll = wm_xr_operator_sessionactive;

  /* flags */
  ot->flag = OPTYPE_UNDO;

  /* properties */
  static const float default_offset[3] = {0};

  RNA_def_boolean(
      ot->srna, "location_lock", false, "Lock Location", "Preserve objects' original location");
  RNA_def_float(ot->srna,
                "location_interpolation",
                0.0f,
                0.0f,
                1.0f,
                "Location Interpolation",
                "Interpolation factor between object and controller locations",
                0.0f,
                1.0f);
  RNA_def_float_translation(ot->srna,
                            "location_offset",
                            3,
                            default_offset,
                            -FLT_MAX,
                            FLT_MAX,
                            "Location Offset",
                            "Additional location offset in controller space",
                            -FLT_MAX,
                            FLT_MAX);
  RNA_def_boolean(
      ot->srna, "rotation_lock", false, "Lock Rotation", "Preserve objects' original rotation");
  RNA_def_float(ot->srna,
                "rotation_interpolation",
                0.0f,
                0.0f,
                1.0f,
                "Rotation Interpolation",
                "Interpolation factor between object and controller rotations",
                0.0f,
                1.0f);
  RNA_def_float_rotation(ot->srna,
                         "rotation_offset",
                         3,
                         default_offset,
                         -2 * M_PI,
                         2 * M_PI,
                         "Rotation Offset",
                         "Additional rotation offset in controller space",
                         -2 * M_PI,
                         2 * M_PI);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name XR Constraints Toggle
 *
 * Toggles enabled/auto key behavior for XR constraint objects.
 * \{ */

static void wm_xr_constraint_toggle(char *flag, bool enable, bool autokey)
{
  if (enable) {
    if ((*flag & XR_OBJECT_ENABLE) != 0) {
      *flag &= ~(XR_OBJECT_ENABLE);
    }
    else {
      *flag |= XR_OBJECT_ENABLE;
    }
  }

  if (autokey) {
    if ((*flag & XR_OBJECT_AUTOKEY) != 0) {
      *flag &= ~(XR_OBJECT_AUTOKEY);
    }
    else {
      *flag |= XR_OBJECT_AUTOKEY;
    }
  }
}

static int wm_xr_constraints_toggle_exec(bContext *C, wmOperator *op)
{
  wmWindowManager *wm = CTX_wm_manager(C);
  XrSessionSettings *settings = &wm->xr.session_settings;
  bool headset, controller0, controller1, enable, autokey;

  PropertyRNA *prop = RNA_struct_find_property(op->ptr, "headset");
  headset = prop ? RNA_property_boolean_get(op->ptr, prop) : true;

  prop = RNA_struct_find_property(op->ptr, "controller0");
  controller0 = prop ? RNA_property_boolean_get(op->ptr, prop) : true;

  prop = RNA_struct_find_property(op->ptr, "controller1");
  controller1 = prop ? RNA_property_boolean_get(op->ptr, prop) : true;

  prop = RNA_struct_find_property(op->ptr, "enable");
  enable = prop ? RNA_property_boolean_get(op->ptr, prop) : true;

  prop = RNA_struct_find_property(op->ptr, "autokey");
  autokey = prop ? RNA_property_boolean_get(op->ptr, prop) : false;

  if (headset) {
    wm_xr_constraint_toggle(&settings->headset_flag, enable, autokey);
  }
  if (controller0) {
    wm_xr_constraint_toggle(&settings->controller0_flag, enable, autokey);
  }
  if (controller1) {
    wm_xr_constraint_toggle(&settings->controller1_flag, enable, autokey);
  }

  WM_event_add_notifier(C, NC_WM | ND_XR_DATA_CHANGED, NULL);

  return OPERATOR_FINISHED;
}

static void WM_OT_xr_constraints_toggle(wmOperatorType *ot)
{
  /* identifiers */
  ot->name = "XR Constraints Toggle";
  ot->idname = "WM_OT_xr_constraints_toggle";
  ot->description = "Toggles enabled/auto key behavior for VR constraint objects";

  /* callbacks */
  ot->exec = wm_xr_constraints_toggle_exec;
  ot->poll = wm_xr_operator_sessionactive;

  /* properties */
  RNA_def_boolean(ot->srna, "headset", true, "Headset", "Toggle behavior for the headset object");
  RNA_def_boolean(ot->srna,
                  "controller0",
                  true,
                  "Controller 0",
                  "Toggle behavior for the first controller object ");
  RNA_def_boolean(ot->srna,
                  "controller1",
                  true,
                  "Controller 1",
                  "Toggle behavior for the second controller object");
  RNA_def_boolean(ot->srna, "enable", true, "Enable", "Toggle constraint enabled behavior");
  RNA_def_boolean(ot->srna, "autokey", false, "Auto Key", "Toggle auto keying behavior");
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Operator Registration
 * \{ */

void wm_xr_operatortypes_register(void)
{
  WM_operatortype_append(WM_OT_xr_session_toggle);
  WM_operatortype_append(WM_OT_xr_select_raycast);
  WM_operatortype_append(WM_OT_xr_grab);
  WM_operatortype_append(WM_OT_xr_constraints_toggle);
}

/** \} */
