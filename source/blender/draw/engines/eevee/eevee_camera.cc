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
 *
 * Copyright 2021, Blender Foundation.
 */

/** \file
 * \ingroup eevee
 */

#include <array>

#include "DRW_render.h"

#include "DNA_camera_types.h"
#include "DNA_view3d_types.h"

#include "BKE_camera.h"
#include "DEG_depsgraph_query.h"
#include "RE_pipeline.h"

#include "eevee_camera.hh"
#include "eevee_instance.hh"

namespace blender::eevee {

/* -------------------------------------------------------------------- */
/** \name eCameraType
 * \{ */

static eCameraType from_camera(const ::Camera *camera)
{
  switch (camera->type) {
    default:
    case CAM_PERSP:
      return CAMERA_PERSP;
    case CAM_ORTHO:
      return CAMERA_ORTHO;
    case CAM_PANO:
      switch (camera->panorama_type) {
        default:
        case CAM_PANO_EQUIRECTANGULAR:
          return CAMERA_PANO_EQUIRECT;
        case CAM_PANO_FISHEYE_EQUIDISTANT:
          return CAMERA_PANO_EQUIDISTANT;
        case CAM_PANO_FISHEYE_EQUISOLID:
          return CAMERA_PANO_EQUISOLID;
        case CAM_PANO_MIRRORBALL:
          return CAMERA_PANO_MIRROR;
      }
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Camera
 * \{ */

void Camera::init(void)
{
  const Object *camera_eval = inst_.camera_eval_object;
  synced_ = false;
  /* Swap! */
  data_id_ = !data_id_;

  CameraData &data = data_[data_id_];

  if (camera_eval) {
    const ::Camera *cam = reinterpret_cast<const ::Camera *>(camera_eval->data);
    data.type = from_camera(cam);
  }
  else if (inst_.drw_view) {
    data.type = DRW_view_is_persp_get(inst_.drw_view) ? CAMERA_PERSP : CAMERA_ORTHO;
  }
  else {
    /* Lightprobe baking. */
    data.type = CAMERA_PERSP;
  }
}

void Camera::sync(void)
{
  const Object *camera_eval = inst_.camera_eval_object;
  CameraData &data = data_[data_id_];

  data.filter_size = inst_.scene->r.gauss;

  if (inst_.drw_view) {
    DRW_view_viewmat_get(inst_.drw_view, data.viewmat, false);
    DRW_view_viewmat_get(inst_.drw_view, data.viewinv, true);
    DRW_view_winmat_get(inst_.drw_view, data.winmat, false);
    DRW_view_winmat_get(inst_.drw_view, data.wininv, true);
    DRW_view_persmat_get(inst_.drw_view, data.persmat, false);
    DRW_view_persmat_get(inst_.drw_view, data.persinv, true);
    DRW_view_camtexco_get(inst_.drw_view, data.uv_scale);
  }
  else if (inst_.render) {
    /* TODO(fclem) Overscan */
    // RE_GetCameraWindowWithOverscan(inst_.render->re, g_data->overscan, data.winmat);
    RE_GetCameraWindow(inst_.render->re, camera_eval, data.winmat);
    RE_GetCameraModelMatrix(inst_.render->re, camera_eval, data.viewinv);
    invert_m4_m4(data.viewmat, data.viewinv);
    invert_m4_m4(data.wininv, data.winmat);
    mul_m4_m4m4(data.persmat, data.winmat, data.viewmat);
    invert_m4_m4(data.persinv, data.persmat);
    data.uv_scale = vec2(1.0f);
    data.uv_bias = vec2(0.0f);
  }
  else {
    unit_m4(data.viewmat);
    unit_m4(data.viewinv);
    perspective_m4(data.winmat, -0.1f, 0.1f, -0.1f, 0.1f, 0.1f, 1.0f);
    invert_m4_m4(data.wininv, data.winmat);
    mul_m4_m4m4(data.persmat, data.winmat, data.viewmat);
    invert_m4_m4(data.persinv, data.persmat);
  }

  if (camera_eval) {
    const ::Camera *cam = reinterpret_cast<const ::Camera *>(camera_eval->data);
    data.clip_near = cam->clip_start;
    data.clip_far = cam->clip_end;
    data.fisheye_fov = cam->fisheye_fov;
    data.fisheye_lens = cam->fisheye_lens;
    data.equirect_bias.x = -cam->longitude_min + M_PI_2;
    data.equirect_bias.y = -cam->latitude_min + M_PI_2;
    data.equirect_scale.x = cam->longitude_min - cam->longitude_max;
    data.equirect_scale.y = cam->latitude_min - cam->latitude_max;
    /* Combine with uv_scale/bias to avoid doing extra computation. */
    data.equirect_bias += data.uv_bias * data.equirect_scale;
    data.equirect_scale *= data.uv_scale;

    data.equirect_scale_inv = 1.0f / data.equirect_scale;
  }
  else if (inst_.drw_view) {
    data.clip_near = DRW_view_near_distance_get(inst_.drw_view);
    data.clip_far = DRW_view_far_distance_get(inst_.drw_view);
    data.fisheye_fov = data.fisheye_lens = -1.0f;
    data.equirect_bias = vec2(0.0f);
    data.equirect_scale = vec2(0.0f);
  }

  data_[data_id_].push_update();

  synced_ = true;

  /* Detect changes in parameters. */
  if (data_[data_id_] != data_[!data_id_]) {
    inst_.sampling.reset();
  }
}

/** \} */

}  // namespace blender::eevee
