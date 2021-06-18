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
 * \name Window-Manager XR Drawing
 *
 * Implements Blender specific drawing functionality for use with the Ghost-XR API.
 */

#include <string.h>

#include "BKE_context.h"

#include "BLI_math.h"

#include "ED_view3d_offscreen.h"

#include "GHOST_C-api.h"

#include "GPU_batch_presets.h"
#include "GPU_immediate.h"
#include "GPU_matrix.h"
#include "GPU_viewport.h"

#include "WM_api.h"

#include "wm_surface.h"
#include "wm_xr_intern.h"

void wm_xr_pose_to_mat(const GHOST_XrPose *pose, float r_mat[4][4])
{
  quat_to_mat4(r_mat, pose->orientation_quat);
  copy_v3_v3(r_mat[3], pose->position);
}

void wm_xr_pose_scale_to_mat(const GHOST_XrPose *pose, float scale, float r_mat[4][4])
{
  wm_xr_pose_to_mat(pose, r_mat);

  BLI_assert(scale > 0.0f);
  mul_v3_fl(r_mat[0], scale);
  mul_v3_fl(r_mat[1], scale);
  mul_v3_fl(r_mat[2], scale);
}

void wm_xr_pose_to_imat(const GHOST_XrPose *pose, float r_imat[4][4])
{
  float iquat[4];
  invert_qt_qt_normalized(iquat, pose->orientation_quat);
  quat_to_mat4(r_imat, iquat);
  translate_m4(r_imat, -pose->position[0], -pose->position[1], -pose->position[2]);
}

void wm_xr_pose_scale_to_imat(const GHOST_XrPose *pose, float scale, float r_imat[4][4])
{
  float iquat[4];
  invert_qt_qt_normalized(iquat, pose->orientation_quat);
  quat_to_mat4(r_imat, iquat);

  BLI_assert(scale > 0.0f);
  scale = 1.0f / scale;
  mul_v3_fl(r_imat[0], scale);
  mul_v3_fl(r_imat[1], scale);
  mul_v3_fl(r_imat[2], scale);

  translate_m4(r_imat, -pose->position[0], -pose->position[1], -pose->position[2]);
}

static void wm_xr_draw_matrices_create(const wmXrDrawData *draw_data,
                                       const GHOST_XrDrawViewInfo *draw_view,
                                       const XrSessionSettings *session_settings,
                                       const wmXrSessionState *session_state,
                                       float r_view_mat[4][4],
                                       float r_proj_mat[4][4])
{
  GHOST_XrPose eye_pose;
  float eye_inv[4][4], base_inv[4][4], nav_inv[4][4], m[4][4];

  /* Calculate inverse eye matrix. */
  copy_qt_qt(eye_pose.orientation_quat, draw_view->eye_pose.orientation_quat);
  copy_v3_v3(eye_pose.position, draw_view->eye_pose.position);
  if ((session_settings->flag & XR_SESSION_USE_POSITION_TRACKING) == 0) {
    sub_v3_v3(eye_pose.position, draw_view->local_pose.position);
  }
  if ((session_settings->flag & XR_SESSION_USE_ABSOLUTE_TRACKING) == 0) {
    sub_v3_v3(eye_pose.position, draw_data->eye_position_ofs);
  }
  wm_xr_pose_to_imat(&eye_pose, eye_inv);

  /* Apply base pose and navigation. */
  wm_xr_pose_scale_to_imat(&draw_data->base_pose, draw_data->base_scale, base_inv);
  wm_xr_pose_scale_to_imat(&session_state->nav_pose, session_state->nav_scale, nav_inv);
  mul_m4_m4m4(m, eye_inv, base_inv);
  mul_m4_m4m4(r_view_mat, m, nav_inv);

  perspective_m4_fov(r_proj_mat,
                     draw_view->fov.angle_left,
                     draw_view->fov.angle_right,
                     draw_view->fov.angle_up,
                     draw_view->fov.angle_down,
                     session_settings->clip_start,
                     session_settings->clip_end);
}

static void wm_xr_draw_viewport_buffers_to_active_framebuffer(
    const wmXrRuntimeData *runtime_data,
    const wmXrSurfaceData *surface_data,
    const GHOST_XrDrawViewInfo *draw_view)
{
  const bool is_upside_down = GHOST_XrSessionNeedsUpsideDownDrawing(runtime_data->context);
  rcti rect = {.xmin = 0, .ymin = 0, .xmax = draw_view->width - 1, .ymax = draw_view->height - 1};

  wmViewport(&rect);

  /* For upside down contexts, draw with inverted y-values. */
  if (is_upside_down) {
    SWAP(int, rect.ymin, rect.ymax);
  }
  GPU_viewport_draw_to_screen_ex(
      surface_data->viewport, 0, &rect, draw_view->expects_srgb_buffer, true);
}

/**
 * \brief Draw a viewport for a single eye.
 *
 * This is the main viewport drawing function for VR sessions. It's assigned to Ghost-XR as a
 * callback (see GHOST_XrDrawViewFunc()) and executed for each view (read: eye).
 */
void wm_xr_draw_view(const GHOST_XrDrawViewInfo *draw_view, void *customdata)
{
  wmXrDrawData *draw_data = customdata;
  wmXrData *xr_data = draw_data->xr_data;
  wmXrSurfaceData *surface_data = draw_data->surface_data;
  wmXrSessionState *session_state = &xr_data->runtime->session_state;
  XrSessionSettings *settings = &xr_data->session_settings;

  const int display_flags = V3D_OFSDRAW_OVERRIDE_SCENE_SETTINGS | settings->draw_flags;

  float viewmat[4][4], winmat[4][4];

  BLI_assert(WM_xr_session_is_ready(xr_data));

  wm_xr_session_draw_data_update(session_state, settings, draw_view, draw_data);
  wm_xr_draw_matrices_create(draw_data, draw_view, settings, session_state, viewmat, winmat);
  wm_xr_session_state_update(settings, draw_data, draw_view, viewmat, session_state);

  if (!wm_xr_session_surface_offscreen_ensure(surface_data, draw_view)) {
    return;
  }

  /* In case a framebuffer is still bound from drawing the last eye. */
  GPU_framebuffer_restore();
  /* Some systems have drawing glitches without this. */
  GPU_clear_depth(1.0f);

  /* Draws the view into the surface_data->viewport's frame-buffers. */
  ED_view3d_draw_offscreen_simple(draw_data->depsgraph,
                                  draw_data->scene,
                                  &settings->shading,
                                  settings->shading.type,
                                  draw_view->width,
                                  draw_view->height,
                                  display_flags,
                                  viewmat,
                                  winmat,
                                  settings->clip_start,
                                  settings->clip_end,
                                  true,
                                  false,
                                  true,
                                  NULL,
                                  false,
                                  surface_data->offscreen,
                                  surface_data->viewport);

  /* The draw-manager uses both GPUOffscreen and GPUViewport to manage frame and texture buffers. A
   * call to GPU_viewport_draw_to_screen() is still needed to get the final result from the
   * viewport buffers composited together and potentially color managed for display on screen.
   * It needs a bound frame-buffer to draw into, for which we simply reuse the GPUOffscreen one.
   *
   * In a next step, Ghost-XR will use the currently bound frame-buffer to retrieve the image
   * to be submitted to the OpenXR swap-chain. So do not un-bind the off-screen yet! */

  GPU_offscreen_bind(surface_data->offscreen, false);

  wm_xr_draw_viewport_buffers_to_active_framebuffer(xr_data->runtime, surface_data, draw_view);
}

void wm_xr_draw_controllers(const bContext *UNUSED(C), ARegion *UNUSED(region), void *customdata)
{
  const wmXrData *xr = customdata;
  const XrSessionSettings *settings = &xr->session_settings;
  const wmXrSessionState *state = &xr->runtime->session_state;

  switch (settings->controller_draw_style) {
    case XR_CONTROLLER_DRAW_AXES: {
      const float r[4] = {1.0f, 0.2f, 0.322f, 1.0f};
      const float g[4] = {0.545f, 0.863f, 0.0f, 1.0f};
      const float b[4] = {0.157f, 0.565f, 1.0f, 1.0f};
      const float scale = 0.1f;
      float x_axis[3], y_axis[3], z_axis[3];

      GPUVertFormat *format = immVertexFormat();
      uint pos = GPU_vertformat_attr_add(format, "pos", GPU_COMP_F32, 3, GPU_FETCH_FLOAT);
      immBindBuiltinProgram(GPU_SHADER_3D_UNIFORM_COLOR);

      GPU_depth_test(GPU_DEPTH_NONE);
      GPU_blend(GPU_BLEND_NONE);
      GPU_line_width(3.0f);

      for (int i = 0; i < 2; ++i) {
        const float(*mat)[4] = state->controllers[i].mat;
        madd_v3_v3v3fl(x_axis, mat[3], mat[0], scale);
        madd_v3_v3v3fl(y_axis, mat[3], mat[1], scale);
        madd_v3_v3v3fl(z_axis, mat[3], mat[2], scale);

        immBegin(GPU_PRIM_LINES, 2);
        immUniformColor4fv(r);
        immVertex3fv(pos, mat[3]);
        immVertex3fv(pos, x_axis);
        immEnd();

        immBegin(GPU_PRIM_LINES, 2);
        immUniformColor4fv(g);
        immVertex3fv(pos, mat[3]);
        immVertex3fv(pos, y_axis);
        immEnd();

        immBegin(GPU_PRIM_LINES, 2);
        immUniformColor4fv(b);
        immVertex3fv(pos, mat[3]);
        immVertex3fv(pos, z_axis);
        immEnd();
      }

      immUnbindProgram();
      break;
    }
    case XR_CONTROLLER_DRAW_RAY: {
      /* Sphere. */
      {
        const float color[4] = {0.422f, 0.438f, 0.446f, 0.4f};
        const float scale = 0.05f;

        GPUBatch *sphere = GPU_batch_preset_sphere(2);
        GPU_batch_program_set_builtin(sphere, GPU_SHADER_3D_UNIFORM_COLOR);
        GPU_batch_uniform_4fv(sphere, "color", color);

        GPU_depth_test(GPU_DEPTH_NONE);
        GPU_blend(GPU_BLEND_ALPHA);

        for (int i = 0; i < 2; ++i) {
          GPU_matrix_push();
          GPU_matrix_mul(state->controllers[i].mat);
          GPU_matrix_scale_1f(scale);
          GPU_batch_draw(sphere);
          GPU_matrix_pop();
        }
      }

      /* Ray. */
      {
        const float color[4] = {0.35f, 0.35f, 1.0f, 0.5f};
        const float scale = settings->clip_end;
        float ray[3];

        GPUVertFormat *format = immVertexFormat();
        uint pos = GPU_vertformat_attr_add(format, "pos", GPU_COMP_F32, 3, GPU_FETCH_FLOAT);
        immBindBuiltinProgram(GPU_SHADER_3D_UNIFORM_COLOR);
        immUniformColor4fv(color);

        GPU_depth_test(GPU_DEPTH_LESS_EQUAL);
        GPU_line_width(3.0f);

        for (int i = 0; i < 2; ++i) {
          const float(*mat)[4] = state->controllers[i].mat;
          madd_v3_v3v3fl(ray, mat[3], mat[2], -scale);

          immBegin(GPU_PRIM_LINES, 2);
          immVertex3fv(pos, mat[3]);
          immVertex3fv(pos, ray);
          immEnd();
        }

        immUnbindProgram();
      }
      break;
    }
  }
}
