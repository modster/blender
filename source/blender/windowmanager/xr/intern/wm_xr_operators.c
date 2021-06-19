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

/** \} */

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
  wm_xr_session_toggle(wm, win, wm_xr_session_update_screen_on_exit_cb);
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
/** \name XR Grab Utilities
 * \{ */

typedef struct XrGrabData {
  float mat_prev[4][4];
  float mat_other_prev[4][4];
  bool bimanual_prev;
} XrGrabData;

static void wm_xr_grab_init(wmOperator *op)
{
  BLI_assert(op->customdata == NULL);

  op->customdata = MEM_callocN(sizeof(XrGrabData), __func__);
}

static void wm_xr_grab_uninit(wmOperator *op)
{
  MEM_SAFE_FREE(op->customdata);
}

static void wm_xr_grab_update(wmOperator *op, const wmXrActionData *actiondata)
{
  XrGrabData *data = op->customdata;

  quat_to_mat4(data->mat_prev, actiondata->controller_rot);
  copy_v3_v3(data->mat_prev[3], actiondata->controller_loc);

  if (actiondata->bimanual) {
    quat_to_mat4(data->mat_other_prev, actiondata->controller_rot_other);
    copy_v3_v3(data->mat_other_prev[3], actiondata->controller_loc_other);
    data->bimanual_prev = true;
  }
  else {
    data->bimanual_prev = false;
  }
}

static void orient_mat_z_normalized(float R[4][4], const float z_axis[3])
{
  const float scale = len_v3(R[0]);
  float x_axis[3], y_axis[3];

  cross_v3_v3v3(y_axis, z_axis, R[0]);
  normalize_v3(y_axis);
  mul_v3_v3fl(R[1], y_axis, scale);

  cross_v3_v3v3(x_axis, R[1], z_axis);
  normalize_v3(x_axis);
  mul_v3_v3fl(R[0], x_axis, scale);

  mul_v3_v3fl(R[2], z_axis, scale);
}

static void wm_xr_grab_navlocks_apply(const float nav_mat[4][4],
                                      const float nav_inv[4][4],
                                      bool loc_lock,
                                      bool locz_lock,
                                      bool rotz_lock,
                                      float r_prev[4][4],
                                      float r_curr[4][4])
{
  /* Locked in base pose coordinates. */
  float prev_base[4][4], curr_base[4][4];

  mul_m4_m4m4(prev_base, nav_inv, r_prev);
  mul_m4_m4m4(curr_base, nav_inv, r_curr);

  if (rotz_lock) {
    const float z_axis[3] = {0.0f, 0.0f, 1.0f};
    orient_mat_z_normalized(prev_base, z_axis);
    orient_mat_z_normalized(curr_base, z_axis);
  }

  if (loc_lock) {
    copy_v3_v3(curr_base[3], prev_base[3]);
  }
  else if (locz_lock) {
    curr_base[3][2] = prev_base[3][2];
  }

  mul_m4_m4m4(r_prev, nav_mat, prev_base);
  mul_m4_m4m4(r_curr, nav_mat, curr_base);
}

static void wm_xr_grab_compute(const wmXrActionData *actiondata,
                               const XrGrabData *data,
                               const Object *obedit,
                               const float nav_mat[4][4],
                               const float nav_inv[4][4],
                               bool reverse,
                               bool loc_lock,
                               bool locz_lock,
                               bool rot_lock,
                               bool rotz_lock,
                               float r_delta[4][4])
{
  const bool nav_lock = (nav_mat && nav_inv);
  float prev[4][4], curr[4][4];

  if (!rot_lock) {
    copy_m4_m4(prev, data->mat_prev);
    zero_v3(prev[3]);
    quat_to_mat4(curr, actiondata->controller_rot);
  }
  else {
    unit_m4(prev);
    unit_m4(curr);
  }

  if (!loc_lock || nav_lock) {
    copy_v3_v3(prev[3], data->mat_prev[3]);
    copy_v3_v3(curr[3], actiondata->controller_loc);
  }

  if (obedit) {
    mul_m4_m4m4(prev, obedit->imat, prev);
    mul_m4_m4m4(curr, obedit->imat, curr);
  }

  if (nav_lock) {
    wm_xr_grab_navlocks_apply(nav_mat, nav_inv, loc_lock, locz_lock, rotz_lock, prev, curr);
  }

  if (reverse) {
    invert_m4(curr);
    mul_m4_m4m4(r_delta, prev, curr);
  }
  else {
    invert_m4(prev);
    mul_m4_m4m4(r_delta, curr, prev);
  }
}

static void wm_xr_grab_compute_bimanual(const wmXrActionData *actiondata,
                                        const XrGrabData *data,
                                        const Object *obedit,
                                        const float nav_mat[4][4],
                                        const float nav_inv[4][4],
                                        bool reverse,
                                        bool loc_lock,
                                        bool locz_lock,
                                        bool rot_lock,
                                        bool rotz_lock,
                                        bool scale_lock,
                                        float r_delta[4][4])
{
  const bool nav_lock = (nav_mat && nav_inv);
  float prev[4][4], curr[4][4];
  unit_m4(prev);
  unit_m4(curr);

  if (!rot_lock) {
    /* Rotation. */
    float x_axis_prev[3], x_axis_curr[3], y_axis_prev[3], y_axis_curr[3], z_axis_prev[3],
        z_axis_curr[3];
    float m0[3][3], m1[3][3];
    quat_to_mat3(m0, actiondata->controller_rot);
    quat_to_mat3(m1, actiondata->controller_rot_other);

    /* x-axis is the base line between the two controllers. */
    sub_v3_v3v3(x_axis_prev, data->mat_prev[3], data->mat_other_prev[3]);
    sub_v3_v3v3(x_axis_curr, actiondata->controller_loc, actiondata->controller_loc_other);
    /* y-axis is the average of the controllers' y-axes. */
    add_v3_v3v3(y_axis_prev, data->mat_prev[1], data->mat_other_prev[1]);
    mul_v3_fl(y_axis_prev, 0.5f);
    add_v3_v3v3(y_axis_curr, m0[1], m1[1]);
    mul_v3_fl(y_axis_curr, 0.5f);
    /* z-axis is the cross product of the two. */
    cross_v3_v3v3(z_axis_prev, x_axis_prev, y_axis_prev);
    cross_v3_v3v3(z_axis_curr, x_axis_curr, y_axis_curr);
    /* Fix the y-axis to be orthogonal. */
    cross_v3_v3v3(y_axis_prev, z_axis_prev, x_axis_prev);
    cross_v3_v3v3(y_axis_curr, z_axis_curr, x_axis_curr);
    /* Normalize. */
    normalize_v3_v3(prev[0], x_axis_prev);
    normalize_v3_v3(prev[1], y_axis_prev);
    normalize_v3_v3(prev[2], z_axis_prev);
    normalize_v3_v3(curr[0], x_axis_curr);
    normalize_v3_v3(curr[1], y_axis_curr);
    normalize_v3_v3(curr[2], z_axis_curr);
  }

  if (!loc_lock || nav_lock) {
    /* Translation: translation of the averaged controller locations. */
    add_v3_v3v3(prev[3], data->mat_prev[3], data->mat_other_prev[3]);
    mul_v3_fl(prev[3], 0.5f);
    add_v3_v3v3(curr[3], actiondata->controller_loc, actiondata->controller_loc_other);
    mul_v3_fl(curr[3], 0.5f);
  }

  if (!scale_lock) {
    /* Scaling: distance between controllers. */
    float scale, v[3];

    sub_v3_v3v3(v, data->mat_prev[3], data->mat_other_prev[3]);
    scale = len_v3(v);
    mul_v3_fl(prev[0], scale);
    mul_v3_fl(prev[1], scale);
    mul_v3_fl(prev[2], scale);

    sub_v3_v3v3(v, actiondata->controller_loc, actiondata->controller_loc_other);
    scale = len_v3(v);
    mul_v3_fl(curr[0], scale);
    mul_v3_fl(curr[1], scale);
    mul_v3_fl(curr[2], scale);
  }

  if (obedit) {
    mul_m4_m4m4(prev, obedit->imat, prev);
    mul_m4_m4m4(curr, obedit->imat, curr);
  }

  if (nav_lock) {
    wm_xr_grab_navlocks_apply(nav_mat, nav_inv, loc_lock, locz_lock, rotz_lock, prev, curr);
  }

  if (reverse) {
    invert_m4(curr);
    mul_m4_m4m4(r_delta, prev, curr);
  }
  else {
    invert_m4(prev);
    mul_m4_m4m4(r_delta, curr, prev);
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name XR Navigation Grab
 *
 * Navigates the scene by grabbing with XR controllers.
 * \{ */

static int wm_xr_navigation_grab_invoke_3d(bContext *C, wmOperator *op, const wmEvent *event)
{
  BLI_assert(event->type == EVT_XR_ACTION);
  BLI_assert(event->custom == EVT_DATA_XR);
  BLI_assert(event->customdata);

  const wmXrActionData *actiondata = event->customdata;

  wm_xr_grab_init(op);
  wm_xr_grab_update(op, actiondata);

  WM_event_add_modal_handler(C, op);

  return OPERATOR_RUNNING_MODAL;
}

static int wm_xr_navigation_grab_exec(bContext *UNUSED(C), wmOperator *UNUSED(op))
{
  return OPERATOR_CANCELLED;
}

static int wm_xr_navigation_grab_modal_3d(bContext *C, wmOperator *op, const wmEvent *event)
{
  BLI_assert(event->type == EVT_XR_ACTION);
  BLI_assert(event->custom == EVT_DATA_XR);
  BLI_assert(event->customdata);

  const wmXrActionData *actiondata = event->customdata;
  XrGrabData *data = op->customdata;
  wmWindowManager *wm = CTX_wm_manager(C);
  wmXrData *xr = &wm->xr;
  bool loc_lock, locz_lock, rot_lock, rotz_lock, scale_lock;
  GHOST_XrPose nav_pose;
  float nav_scale, nav_mat[4][4], nav_inv[4][4], delta[4][4], m[4][4];

  PropertyRNA *prop = RNA_struct_find_property(op->ptr, "lock_location");
  loc_lock = prop ? RNA_property_boolean_get(op->ptr, prop) : false;
  prop = RNA_struct_find_property(op->ptr, "lock_location_z");
  locz_lock = prop ? RNA_property_boolean_get(op->ptr, prop) : false;
  prop = RNA_struct_find_property(op->ptr, "lock_rotation");
  rot_lock = prop ? RNA_property_boolean_get(op->ptr, prop) : false;
  prop = RNA_struct_find_property(op->ptr, "lock_rotation_z");
  rotz_lock = prop ? RNA_property_boolean_get(op->ptr, prop) : false;
  prop = RNA_struct_find_property(op->ptr, "lock_scale");
  scale_lock = prop ? RNA_property_boolean_get(op->ptr, prop) : false;

  const bool do_bimanual = (actiondata->bimanual && data->bimanual_prev);
  const bool apply_navigation = (do_bimanual ? !(loc_lock && rot_lock && scale_lock) :
                                               !(loc_lock && rot_lock)) &&
                                (actiondata->bimanual || !data->bimanual_prev);

  if (apply_navigation) {
    const bool nav_lock = (loc_lock || locz_lock || rotz_lock);

    WM_xr_session_state_nav_location_get(xr, nav_pose.position);
    WM_xr_session_state_nav_rotation_get(xr, nav_pose.orientation_quat);
    WM_xr_session_state_nav_scale_get(xr, &nav_scale);

    wm_xr_pose_scale_to_mat(&nav_pose, nav_scale, nav_mat);
    if (nav_lock) {
      wm_xr_pose_scale_to_imat(&nav_pose, nav_scale, nav_inv);
    }

    if (do_bimanual) {
      wm_xr_grab_compute_bimanual(actiondata,
                                  data,
                                  NULL,
                                  nav_lock ? nav_mat : NULL,
                                  nav_lock ? nav_inv : NULL,
                                  true,
                                  loc_lock,
                                  locz_lock,
                                  rot_lock,
                                  rotz_lock,
                                  scale_lock,
                                  delta);
    }
    else {
      wm_xr_grab_compute(actiondata,
                         data,
                         NULL,
                         nav_lock ? nav_mat : NULL,
                         nav_lock ? nav_inv : NULL,
                         true,
                         loc_lock,
                         locz_lock,
                         rot_lock,
                         rotz_lock,
                         delta);
    }

    mul_m4_m4m4(m, delta, nav_mat);

    /* Limit scale to reasonable values. */
    nav_scale = len_v3(m[0]);

    if (!(nav_scale < 0.001f || nav_scale > 1000.0f)) {
      WM_xr_session_state_nav_location_set(xr, m[3]);
      if (!rot_lock) {
        mat4_to_quat(nav_pose.orientation_quat, m);
        normalize_qt(nav_pose.orientation_quat);
        WM_xr_session_state_nav_rotation_set(xr, nav_pose.orientation_quat);
      }
      if (!scale_lock && do_bimanual) {
        WM_xr_session_state_nav_scale_set(xr, nav_scale);
      }
    }
  }

  if (actiondata->bimanual) {
    if (!data->bimanual_prev) {
      quat_to_mat4(data->mat_prev, actiondata->controller_rot);
      copy_v3_v3(data->mat_prev[3], actiondata->controller_loc);
      quat_to_mat4(data->mat_other_prev, actiondata->controller_rot_other);
      copy_v3_v3(data->mat_other_prev[3], actiondata->controller_loc_other);
    }
    data->bimanual_prev = true;
  }
  else {
    if (data->bimanual_prev) {
      quat_to_mat4(data->mat_prev, actiondata->controller_rot);
      copy_v3_v3(data->mat_prev[3], actiondata->controller_loc);
    }
    data->bimanual_prev = false;
  }

  if (event->val == KM_PRESS) {
    return OPERATOR_RUNNING_MODAL;
  }
  else if (event->val == KM_RELEASE) {
    wm_xr_grab_uninit(op);
    return OPERATOR_FINISHED;
  }

  /* XR events currently only support press and release. */
  BLI_assert(false);
  wm_xr_grab_uninit(op);
  return OPERATOR_CANCELLED;
}

static void WM_OT_xr_navigation_grab(wmOperatorType *ot)
{
  /* identifiers */
  ot->name = "XR Navigation Grab";
  ot->idname = "WM_OT_xr_navigation_grab";
  ot->description = "Navigate the VR scene by grabbing with controllers";

  /* callbacks */
  ot->invoke_3d = wm_xr_navigation_grab_invoke_3d;
  ot->exec = wm_xr_navigation_grab_exec;
  ot->modal_3d = wm_xr_navigation_grab_modal_3d;
  ot->poll = wm_xr_operator_sessionactive;

  /* properties */
  RNA_def_boolean(
      ot->srna, "lock_location", false, "Lock Location", "Prevent changes to viewer location");
  RNA_def_boolean(
      ot->srna, "lock_location_z", false, "Lock Elevation", "Prevent changes to viewer elevation");
  RNA_def_boolean(
      ot->srna, "lock_rotation", false, "Lock Rotation", "Prevent changes to viewer rotation");
  RNA_def_boolean(ot->srna,
                  "lock_rotation_z",
                  false,
                  "Lock Up Orientation",
                  "Prevent changes to viewer up orientation");
  RNA_def_boolean(ot->srna, "lock_scale", false, "Lock Scale", "Prevent changes to viewer scale");
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name XR Raycast Utilities
 * \{ */

static const float g_xr_default_raycast_axis[3] = {0.0f, 0.0f, -1.0f};
static const float g_xr_default_raycast_color[4] = {0.35f, 0.35f, 1.0f, 1.0f};

typedef struct XrRaycastData {
  float origin[3];
  float direction[3];
  float end[3];
  float color[4];
  void *draw_handle;
} XrRaycastData;

static void wm_xr_raycast_draw(const bContext *UNUSED(C),
                               ARegion *UNUSED(region),
                               void *customdata)
{
  const XrRaycastData *data = customdata;

  GPUVertFormat *format = immVertexFormat();
  uint pos = GPU_vertformat_attr_add(format, "pos", GPU_COMP_F32, 3, GPU_FETCH_FLOAT);
  immBindBuiltinProgram(GPU_SHADER_3D_UNIFORM_COLOR);
  immUniformColor4fv(data->color);

  GPU_depth_test(GPU_DEPTH_LESS_EQUAL);
  GPU_line_width(3.0f);

  immBegin(GPU_PRIM_LINES, 2);
  immVertex3fv(pos, data->origin);
  immVertex3fv(pos, data->end);
  immEnd();

  immUnbindProgram();
}

static void wm_xr_raycast_init(wmOperator *op)
{
  BLI_assert(op->customdata == NULL);

  op->customdata = MEM_callocN(sizeof(XrRaycastData), __func__);

  SpaceType *st = BKE_spacetype_from_id(SPACE_VIEW3D);
  if (st) {
    ARegionType *art = BKE_regiontype_from_id(st, RGN_TYPE_XR);
    if (art) {
      ((XrRaycastData *)op->customdata)->draw_handle = ED_region_draw_cb_activate(
          art, wm_xr_raycast_draw, op->customdata, REGION_DRAW_POST_VIEW);
    }
  }
}

static void wm_xr_raycast_uninit(wmOperator *op)
{
  if (op->customdata) {
    SpaceType *st = BKE_spacetype_from_id(SPACE_VIEW3D);
    if (st) {
      ARegionType *art = BKE_regiontype_from_id(st, RGN_TYPE_XR);
      if (art) {
        ED_region_draw_cb_exit(art, ((XrRaycastData *)op->customdata)->draw_handle);
      }
    }

    MEM_freeN(op->customdata);
  }
}

static void wm_xr_raycast_update(wmOperator *op,
                                 const wmXrData *xr,
                                 const wmXrActionData *actiondata)
{
  XrRaycastData *data = op->customdata;
  float axis[3];

  PropertyRNA *prop = RNA_struct_find_property(op->ptr, "axis");
  if (prop) {
    RNA_property_float_get_array(op->ptr, prop, axis);
    normalize_v3(axis);
  }
  else {
    copy_v3_v3(axis, g_xr_default_raycast_axis);
  }

  prop = RNA_struct_find_property(op->ptr, "color");
  if (prop) {
    RNA_property_float_get_array(op->ptr, prop, data->color);
  }
  else {
    copy_v4_v4(data->color, g_xr_default_raycast_color);
  }

  copy_v3_v3(data->origin, actiondata->controller_loc);

  mul_qt_v3(actiondata->controller_rot, axis);
  copy_v3_v3(data->direction, axis);

  mul_v3_v3fl(data->end, data->direction, xr->session_settings.clip_end);
  add_v3_v3(data->end, data->origin);
}

static void wm_xr_raycast(Depsgraph *depsgraph,
                          ViewContext *vc,
                          const float origin[3],
                          const float direction[3],
                          float *ray_dist,
                          bool selectable_only,
                          float r_location[3],
                          float r_normal[3],
                          int *r_index,
                          Object **r_ob,
                          float r_obmat[4][4])
{
  /* Uses same raycast method as Scene.ray_cast(). */
  SnapObjectContext *sctx = ED_transform_snap_object_context_create(vc->scene, 0);

  ED_transform_snap_object_project_ray_ex(
      sctx,
      depsgraph,
      &(const struct SnapObjectParams){
          .snap_select = vc->em ? SNAP_SELECTED : (selectable_only ? SNAP_SELECTABLE : SNAP_ALL)},
      origin,
      direction,
      ray_dist,
      r_location,
      r_normal,
      r_index,
      r_ob,
      r_obmat);

  ED_transform_snap_object_context_destroy(sctx);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name XR Navigation Teleport
 *
 * Casts a ray from an XR controller's pose and teleports to any hit geometry.
 * \{ */

static void wm_xr_navigation_teleport(bContext *C,
                                      wmXrData *xr,
                                      const float origin[3],
                                      const float direction[3],
                                      float *ray_dist,
                                      bool selectable_only,
                                      const bool teleport_axes[3],
                                      float teleport_t)
{
  Depsgraph *depsgraph = CTX_data_ensure_evaluated_depsgraph(C);
  ViewContext vc;
  ED_view3d_viewcontext_init(C, &vc, depsgraph);
  vc.em = NULL; /* Set to NULL to always enable teleport to non-edited objects. */

  float location[3];
  float normal[3];
  int index;
  Object *ob = NULL;
  float obmat[4][4];

  wm_xr_raycast(depsgraph,
                &vc,
                origin,
                direction,
                ray_dist,
                selectable_only,
                location,
                normal,
                &index,
                &ob,
                obmat);

  /* Teleport. */
  if (ob) {
    float nav_location[3], nav_rotation[4], viewer_location[3];
    float nav_axes[3][3], projected[3], v0[3], v1[3];
    float out[3] = {0.0f, 0.0f, 0.0f};

    WM_xr_session_state_nav_location_get(xr, nav_location);
    WM_xr_session_state_nav_rotation_get(xr, nav_rotation);
    WM_xr_session_state_viewer_pose_location_get(xr, viewer_location);

    quat_to_mat3(nav_axes, nav_rotation);

    /* Project locations onto navigation axes. */
    for (int a = 0; a < 3; ++a) {
      normalize_v3(nav_axes[a]);
      project_v3_v3v3_normalized(projected, nav_location, nav_axes[a]);
      if (teleport_axes[a]) {
        /* Interpolate between projected locations. */
        project_v3_v3v3_normalized(v0, location, nav_axes[a]);
        project_v3_v3v3_normalized(v1, viewer_location, nav_axes[a]);
        sub_v3_v3(v0, v1);
        madd_v3_v3fl(projected, v0, teleport_t);
      }
      /* Add to final location. */
      add_v3_v3(out, projected);
    }

    WM_xr_session_state_nav_location_set(xr, out);
  }
}

static int wm_xr_navigation_teleport_invoke_3d(bContext *C, wmOperator *op, const wmEvent *event)
{
  BLI_assert(event->type == EVT_XR_ACTION);
  BLI_assert(event->custom == EVT_DATA_XR);
  BLI_assert(event->customdata);

  wm_xr_raycast_init(op);

  int retval = op->type->modal_3d(C, op, event);

  if ((retval & OPERATOR_RUNNING_MODAL) != 0) {
    WM_event_add_modal_handler(C, op);
  }

  return retval;
}

static int wm_xr_navigation_teleport_exec(bContext *UNUSED(C), wmOperator *UNUSED(op))
{
  return OPERATOR_CANCELLED;
}

static int wm_xr_navigation_teleport_modal_3d(bContext *C, wmOperator *op, const wmEvent *event)
{
  BLI_assert(event->type == EVT_XR_ACTION);
  BLI_assert(event->custom == EVT_DATA_XR);
  BLI_assert(event->customdata);

  const wmXrActionData *actiondata = event->customdata;
  wmWindowManager *wm = CTX_wm_manager(C);
  wmXrData *xr = &wm->xr;

  wm_xr_raycast_update(op, xr, actiondata);

  if (event->val == KM_PRESS) {
    return OPERATOR_RUNNING_MODAL;
  }
  else if (event->val == KM_RELEASE) {
    XrRaycastData *data = op->customdata;
    bool teleport_axes[3];
    float teleport_t, ray_dist;
    bool selectable_only;

    PropertyRNA *prop = RNA_struct_find_property(op->ptr, "teleport_axes");
    if (prop) {
      RNA_property_boolean_get_array(op->ptr, prop, teleport_axes);
    }
    else {
      teleport_axes[0] = teleport_axes[1] = teleport_axes[2] = true;
    }

    prop = RNA_struct_find_property(op->ptr, "interpolation");
    teleport_t = prop ? RNA_property_float_get(op->ptr, prop) : 1.0f;

    prop = RNA_struct_find_property(op->ptr, "selectable_only");
    selectable_only = prop ? RNA_property_boolean_get(op->ptr, prop) : true;

    prop = RNA_struct_find_property(op->ptr, "distance");
    ray_dist = prop ? RNA_property_float_get(op->ptr, prop) : BVH_RAYCAST_DIST_MAX;

    wm_xr_navigation_teleport(C,
                              xr,
                              data->origin,
                              data->direction,
                              &ray_dist,
                              selectable_only,
                              teleport_axes,
                              teleport_t);

    wm_xr_raycast_uninit(op);

    return OPERATOR_FINISHED;
  }

  /* XR events currently only support press and release. */
  BLI_assert(false);
  wm_xr_raycast_uninit(op);
  return OPERATOR_CANCELLED;
}

static void WM_OT_xr_navigation_teleport(wmOperatorType *ot)
{
  /* identifiers */
  ot->name = "XR Navigation Teleport";
  ot->idname = "WM_OT_xr_navigation_teleport";
  ot->description = "Set VR viewer location to controller raycast hit location";

  /* callbacks */
  ot->invoke_3d = wm_xr_navigation_teleport_invoke_3d;
  ot->exec = wm_xr_navigation_teleport_exec;
  ot->modal_3d = wm_xr_navigation_teleport_modal_3d;
  ot->poll = wm_xr_operator_sessionactive;

  /* properties */
  static bool default_teleport_axes[3] = {true, true, true};

  RNA_def_boolean_vector(ot->srna,
                         "teleport_axes",
                         3,
                         default_teleport_axes,
                         "Teleport Axes",
                         "Enabled teleport axes in viewer space");
  RNA_def_float(ot->srna,
                "interpolation",
                1.0f,
                0.0f,
                1.0f,
                "Interpolation",
                "Interpolation factor between viewer and hit locations",
                0.0f,
                1.0f);
  RNA_def_boolean(ot->srna,
                  "selectable_only",
                  true,
                  "Selectable Only",
                  "Only allow selectable objects to influence raycast result");
  RNA_def_float(ot->srna,
                "distance",
                BVH_RAYCAST_DIST_MAX,
                0.0,
                BVH_RAYCAST_DIST_MAX,
                "",
                "Maximum raycast distance",
                0.0,
                BVH_RAYCAST_DIST_MAX);
  RNA_def_float_vector(ot->srna,
                       "axis",
                       3,
                       g_xr_default_raycast_axis,
                       -1.0f,
                       1.0f,
                       "Axis",
                       "Raycast axis in controller space",
                       -1.0f,
                       1.0f);
  RNA_def_float_color(ot->srna,
                      "color",
                      4,
                      g_xr_default_raycast_color,
                      0.0f,
                      1.0f,
                      "Color",
                      "Raycast color",
                      0.0f,
                      1.0f);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name XR Raycast Select
 *
 * Casts a ray from an XR controller's pose and selects any hit geometry.
 * \{ */

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
                                 bool selectable_only,
                                 eSelectOp select_op,
                                 bool deselect_all)
{
  Depsgraph *depsgraph = CTX_data_ensure_evaluated_depsgraph(C);
  ViewContext vc;
  ED_view3d_viewcontext_init(C, &vc, depsgraph);
  vc.em = (vc.obedit && (vc.obedit->type == OB_MESH)) ? BKE_editmesh_from_object(vc.obedit) : NULL;

  float location[3];
  float normal[3];
  int index;
  Object *ob = NULL;
  float obmat[4][4];

  wm_xr_raycast(depsgraph,
                &vc,
                origin,
                direction,
                ray_dist,
                selectable_only,
                location,
                normal,
                &index,
                &ob,
                obmat);

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

  wm_xr_raycast_init(op);

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

  const wmXrActionData *actiondata = event->customdata;
  wmWindowManager *wm = CTX_wm_manager(C);
  wmXrData *xr = &wm->xr;

  wm_xr_raycast_update(op, xr, actiondata);

  if (event->val == KM_PRESS) {
    return OPERATOR_RUNNING_MODAL;
  }
  else if (event->val == KM_RELEASE) {
    XrRaycastData *data = op->customdata;
    eSelectOp select_op = SEL_OP_SET;
    bool deselect_all, selectable_only;
    float ray_dist;

    PropertyRNA *prop = RNA_struct_find_property(op->ptr, "toggle");
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

    prop = RNA_struct_find_property(op->ptr, "distance");
    ray_dist = prop ? RNA_property_float_get(op->ptr, prop) : BVH_RAYCAST_DIST_MAX;

    bool changed = wm_xr_select_raycast(
        C, data->origin, data->direction, &ray_dist, selectable_only, select_op, deselect_all);

    wm_xr_raycast_uninit(op);

    return changed ? OPERATOR_FINISHED : OPERATOR_CANCELLED;
  }

  /* XR events currently only support press and release. */
  BLI_assert(false);
  wm_xr_raycast_uninit(op);
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
  WM_operator_properties_mouse_select(ot);

  /* Override "deselect_all" default value. */
  PropertyRNA *prop = RNA_struct_type_find_property(ot->srna, "deselect_all");
  BLI_assert(prop != NULL);
  RNA_def_property_boolean_default(prop, true);

  RNA_def_boolean(ot->srna,
                  "selectable_only",
                  true,
                  "Selectable Only",
                  "Only allow selectable objects to influence raycast result");
  RNA_def_float(ot->srna,
                "distance",
                BVH_RAYCAST_DIST_MAX,
                0.0,
                BVH_RAYCAST_DIST_MAX,
                "",
                "Maximum raycast distance",
                0.0,
                BVH_RAYCAST_DIST_MAX);
  RNA_def_float_vector(ot->srna,
                       "axis",
                       3,
                       g_xr_default_raycast_axis,
                       -1.0f,
                       1.0f,
                       "Axis",
                       "Raycast axis in controller space",
                       -1.0f,
                       1.0f);
  RNA_def_float_color(ot->srna,
                      "color",
                      4,
                      g_xr_default_raycast_color,
                      0.0f,
                      1.0f,
                      "Color",
                      "Raycast color",
                      0.0f,
                      1.0f);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name XR Transform Grab
 *
 * Transforms selected objects relative to an XR controller's pose.
 * \{ */

static int wm_xr_transform_grab_invoke_3d(bContext *C, wmOperator *op, const wmEvent *event)
{
  BLI_assert(event->type == EVT_XR_ACTION);
  BLI_assert(event->custom == EVT_DATA_XR);
  BLI_assert(event->customdata);

  bool loc_lock, rot_lock, scale_lock;
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
      float eul[3];
      RNA_property_float_get_array(op->ptr, prop, eul);
      eul_to_quat(rot_ofs, eul);
      normalize_qt(rot_ofs);
      rot_ofs_set = true;
    }
  }

  prop = RNA_struct_find_property(op->ptr, "scale_lock");
  scale_lock = prop ? RNA_property_boolean_get(op->ptr, prop) : true;

  if (loc_lock && rot_lock && scale_lock) {
    return OPERATOR_CANCELLED;
  }

  const wmXrActionData *actiondata = event->customdata;
  Object *obedit = CTX_data_edit_object(C);
  BMEditMesh *em = (obedit && (obedit->type == OB_MESH)) ? BKE_editmesh_from_object(obedit) : NULL;
  float q0[4], q1[4], q2[4];
  bool selected = false;

  if (loc_ofs_set) {
    /* Convert to controller space. */
    float m[3][3];
    quat_to_mat3(m, actiondata->controller_rot);
    mul_m3_v3(m, loc_ofs);
  }
  if (rot_ofs_set) {
    /* Convert to controller space. */
    invert_qt_qt_normalized(q0, rot_ofs);
    mul_qt_qtqt(rot_ofs, actiondata->controller_rot, q0);
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
          eul_to_quat(q1, ob->rot);
          interp_qt_qtqt(q0, q1, actiondata->controller_rot, rot_t);
          if (!rot_ofs_set) {
            quat_to_eul(ob->rot, q0);
          }
          update = true;
        }
        else if (rot_ofs_set) {
          eul_to_quat(q0, ob->rot);
        }
        if (rot_ofs_set) {
          rotation_between_quats_to_quat(q1, rot_ofs, q0);
          mul_qt_qtqt(q0, rot_ofs, q1);
          normalize_qt(q0);
          mul_qt_qtqt(q2, actiondata->controller_rot, q1);
          normalize_qt(q2);
          rotation_between_quats_to_quat(q1, q0, q2);

          mul_qt_qtqt(q2, q0, q1);
          normalize_qt(q2);
          quat_to_eul(ob->rot, q2);
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
    return OPERATOR_CANCELLED;
  }

  wm_xr_grab_init(op);
  wm_xr_grab_update(op, actiondata);

  WM_event_add_modal_handler(C, op);

  return OPERATOR_RUNNING_MODAL;
}

static int wm_xr_transform_grab_exec(bContext *UNUSED(C), wmOperator *UNUSED(op))
{
  return OPERATOR_CANCELLED;
}

static int wm_xr_transform_grab_modal_3d(bContext *C, wmOperator *op, const wmEvent *event)
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
  bool loc_lock, rot_lock, scale_lock;
  bool selected = false;
  float delta[4][4], m[4][4];

  PropertyRNA *prop = RNA_struct_find_property(op->ptr, "location_lock");
  loc_lock = prop ? RNA_property_boolean_get(op->ptr, prop) : false;
  prop = RNA_struct_find_property(op->ptr, "rotation_lock");
  rot_lock = prop ? RNA_property_boolean_get(op->ptr, prop) : false;
  prop = RNA_struct_find_property(op->ptr, "scale_lock");
  scale_lock = prop ? RNA_property_boolean_get(op->ptr, prop) : false;

  const bool do_bimanual = (actiondata->bimanual && data->bimanual_prev);
  const bool apply_transform = do_bimanual ? !(loc_lock && rot_lock && scale_lock) :
                                             !(loc_lock && rot_lock);

  if (em) { /* TODO_XR: Non-mesh objects. */
    if (apply_transform) {
      ToolSettings *ts = scene->toolsettings;
      BMesh *bm = em->bm;
      BMIter iter;

      if (do_bimanual) {
        wm_xr_grab_compute_bimanual(actiondata,
                                    data,
                                    obedit,
                                    NULL,
                                    NULL,
                                    false,
                                    loc_lock,
                                    false,
                                    rot_lock,
                                    false,
                                    scale_lock,
                                    delta);
      }
      else {
        wm_xr_grab_compute(
            actiondata, data, obedit, NULL, NULL, false, loc_lock, false, rot_lock, false, delta);
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

    selected = true;
  }
  else {
    if (apply_transform) {
      if (do_bimanual) {
        wm_xr_grab_compute_bimanual(actiondata,
                                    data,
                                    NULL,
                                    NULL,
                                    NULL,
                                    false,
                                    loc_lock,
                                    false,
                                    rot_lock,
                                    false,
                                    scale_lock,
                                    delta);
      }
      else {
        wm_xr_grab_compute(
            actiondata, data, NULL, NULL, NULL, false, loc_lock, false, rot_lock, false, delta);
      }
    }

    CTX_DATA_BEGIN (C, Object *, ob, selected_objects) {
      if (apply_transform) {
        mul_m4_m4m4(m, delta, ob->obmat);

        if (!loc_lock) {
          copy_v3_v3(ob->loc, m[3]);
        }
        if (!rot_lock) {
          mat4_to_eul(ob->rot, m);
        }
        if (!scale_lock && do_bimanual) {
          mat4_to_size(ob->scale, m);
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

  wm_xr_grab_update(op, actiondata);

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

static void WM_OT_xr_transform_grab(wmOperatorType *ot)
{
  /* identifiers */
  ot->name = "XR Transform Grab";
  ot->idname = "WM_OT_xr_transform_grab";
  ot->description = "Transform selected objects relative to a VR controller's pose";

  /* callbacks */
  ot->invoke_3d = wm_xr_transform_grab_invoke_3d;
  ot->exec = wm_xr_transform_grab_exec;
  ot->modal_3d = wm_xr_transform_grab_modal_3d;
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
  RNA_def_boolean(ot->srna, "scale_lock", false, "Lock Scale", "Preserve objects' original scale");
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
  ot->description = "Toggle enabled/auto key behavior for VR constraint objects";

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
  WM_operatortype_append(WM_OT_xr_navigation_grab);
  WM_operatortype_append(WM_OT_xr_navigation_teleport);
  WM_operatortype_append(WM_OT_xr_select_raycast);
  WM_operatortype_append(WM_OT_xr_transform_grab);
  WM_operatortype_append(WM_OT_xr_constraints_toggle);
}

/** \} */
