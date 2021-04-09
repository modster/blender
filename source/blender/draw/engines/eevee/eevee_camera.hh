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

#pragma once

#include <array>

#include "DRW_render.h"

#include "BKE_camera.h"

#include "RE_pipeline.h"

#include "DNA_camera_types.h"
#include "DNA_object_types.h"
#include "DNA_scene_types.h"
#include "DNA_view3d_types.h"

#include "eevee_sampling.hh"
#include "eevee_shader_shared.hh"

namespace blender::eevee {

/* TODO(fclem) Might want to move to eevee_shader_shared.hh. */
static const float cubeface_mat[6][4][4] = {
    /* Pos X */
    {{0.0f, 0.0f, -1.0f, 0.0f},
     {0.0f, -1.0f, 0.0f, 0.0f},
     {-1.0f, 0.0f, 0.0f, 0.0f},
     {0.0f, 0.0f, 0.0f, 1.0f}},
    /* Neg X */
    {{0.0f, 0.0f, 1.0f, 0.0f},
     {0.0f, -1.0f, 0.0f, 0.0f},
     {1.0f, 0.0f, 0.0f, 0.0f},
     {0.0f, 0.0f, 0.0f, 1.0f}},
    /* Pos Y */
    {{1.0f, 0.0f, 0.0f, 0.0f},
     {0.0f, 0.0f, -1.0f, 0.0f},
     {0.0f, 1.0f, 0.0f, 0.0f},
     {0.0f, 0.0f, 0.0f, 1.0f}},
    /* Neg Y */
    {{1.0f, 0.0f, 0.0f, 0.0f},
     {0.0f, 0.0f, 1.0f, 0.0f},
     {0.0f, -1.0f, 0.0f, 0.0f},
     {0.0f, 0.0f, 0.0f, 1.0f}},
    /* Pos Z */
    {{1.0f, 0.0f, 0.0f, 0.0f},
     {0.0f, -1.0f, 0.0f, 0.0f},
     {0.0f, 0.0f, -1.0f, 0.0f},
     {0.0f, 0.0f, 0.0f, 1.0f}},
    /* Neg Z */
    {{-1.0f, 0.0f, 0.0f, 0.0f},
     {0.0f, -1.0f, 0.0f, 0.0f},
     {0.0f, 0.0f, 1.0f, 0.0f},
     {0.0f, 0.0f, 0.0f, 1.0f}},
};

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
/** \name CameraData operators
 * \{ */

inline bool operator==(const CameraData &a, const CameraData &b)
{
  return compare_m4m4(a.persmat, b.persmat, FLT_MIN) && equals_v2v2(a.uv_scale, b.uv_scale) &&
         equals_v2v2(a.uv_bias, b.uv_bias) && equals_v2v2(a.equirect_scale, b.equirect_scale) &&
         equals_v2v2(a.equirect_bias, b.equirect_bias) && (a.fisheye_fov == b.fisheye_fov) &&
         (a.fisheye_lens == b.fisheye_lens) && (a.filter_size == b.filter_size) &&
         (a.type == b.type);
}

inline bool operator!=(const CameraData &a, const CameraData &b)
{
  return !(a == b);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Camera
 * \{ */

class Camera {
 private:
  eCameraType type_;
  /** Random module to know what jitter to apply to the view. */
  Sampling &sampling_;
  /** Double buffered to detect changes and have history for re-projection. */
  struct {
    CameraData *data_;
    GPUUniformBuf *ubo_;
  } current, previous;
  /** True if camera matrix has change since last init. */
  bool has_changed_ = true;
  /** Detects wrong usage. */
  bool synced_ = false;
  /** Last sample we synced with. Avoid double sync. */
  uint64_t last_sample_ = 0;
  /** Original object of the camera. */
  Object *camera_original_ = nullptr;
  /** Evaluated camera object. Only valid after sync. */
  const Object *object_eval_ = nullptr;
  /** Depsgraph to query evaluated camera. */
  const Depsgraph *depsgraph_ = nullptr;
  /** Copy of instance.render_. */
  const RenderEngine *engine_ = nullptr;

 public:
  Camera(Sampling &sampling) : sampling_(sampling)
  {
    current.data_ = (CameraData *)MEM_callocN(sizeof(CameraData), "CameraData");
    current.ubo_ = GPU_uniformbuf_create_ex(sizeof(CameraData), nullptr, "CameraData");

    previous.data_ = (CameraData *)MEM_callocN(sizeof(CameraData), "CameraData");
    previous.ubo_ = GPU_uniformbuf_create_ex(sizeof(CameraData), nullptr, "CameraData");
  };

  ~Camera()
  {
    MEM_SAFE_FREE(current.data_);
    MEM_SAFE_FREE(previous.data_);
    DRW_UBO_FREE_SAFE(current.ubo_);
    DRW_UBO_FREE_SAFE(previous.ubo_);
  };

  void init(const RenderEngine *engine,
            const Depsgraph *depsgraph,
            Object *camera_original,
            const DRWView *drw_view)
  {
    engine_ = engine;
    const Object *camera_eval = DEG_get_evaluated_object(depsgraph, camera_original);
    camera_original_ = camera_original;
    depsgraph_ = depsgraph;
    synced_ = false;

    SWAP(CameraData *, current.data_, previous.data_);
    SWAP(GPUUniformBuf *, current.ubo_, previous.ubo_);

    CameraData &data = *current.data_;

    if (camera_eval) {
      const ::Camera *cam = reinterpret_cast<const ::Camera *>(camera_eval->data);
      data.type = from_camera(cam);
    }
    else {
      data.type = DRW_view_is_persp_get(drw_view) ? CAMERA_PERSP : CAMERA_ORTHO;
    }

    /* Sync early to detect changes. This is ok since we avoid double sync later. */
    this->sync(drw_view);

    /* Detect changes in parameters. */
    has_changed_ = *current.data_ != *previous.data_;
    if (has_changed_) {
      sampling_.reset();
    }
  }

  void sync(const DRWView *drw_view)
  {
    const Scene *scene = DEG_get_evaluated_scene(depsgraph_);
    object_eval_ = DEG_get_evaluated_object(depsgraph_, camera_original_);

    uint64_t sample = sampling_.sample_get();
    if (last_sample_ != sample || !synced_) {
      last_sample_ = sample;
    }
    else {
      /* Avoid double sync. */
      return;
    }

    CameraData &data = *current.data_;

    data.filter_size = scene->r.gauss;

    if (drw_view) {
      DRW_view_viewmat_get(drw_view, data.viewmat, false);
      DRW_view_viewmat_get(drw_view, data.viewinv, true);
      DRW_view_winmat_get(drw_view, data.winmat, false);
      DRW_view_winmat_get(drw_view, data.wininv, true);
      DRW_view_persmat_get(drw_view, data.persmat, false);
      DRW_view_persmat_get(drw_view, data.persinv, true);
      DRW_view_camtexco_get(drw_view, &data.uv_scale[0]);
    }
    else if (engine_) {
      /* TODO(fclem) Overscan */
      // RE_GetCameraWindowWithOverscan(engine_->re, g_data->overscan, data.winmat);
      RE_GetCameraWindow(engine_->re, object_eval_, data.winmat);
      RE_GetCameraModelMatrix(engine_->re, object_eval_, data.viewinv);
      invert_m4_m4(data.viewmat, data.viewinv);
      invert_m4_m4(data.wininv, data.winmat);
      mul_m4_m4m4(data.persmat, data.winmat, data.viewmat);
      invert_m4_m4(data.persinv, data.persmat);
      copy_v2_fl(data.uv_scale, 1.0f);
      copy_v2_fl(data.uv_bias, 0.0f);
    }
    else {
      BLI_assert(0);
    }

    if (object_eval_) {
      const ::Camera *cam = reinterpret_cast<const ::Camera *>(object_eval_->data);
      data.clip_near = cam->clip_start;
      data.clip_far = cam->clip_end;
      data.fisheye_fov = cam->fisheye_fov;
      data.fisheye_lens = cam->fisheye_lens;
      data.equirect_bias[0] = -cam->longitude_min + M_PI_2;
      data.equirect_bias[1] = -cam->latitude_min + M_PI_2;
      data.equirect_scale[0] = cam->longitude_min - cam->longitude_max;
      data.equirect_scale[1] = cam->latitude_min - cam->latitude_max;
      /* Combine with uv_scale/bias to avoid doing extra computation. */
      madd_v2_v2v2(data.equirect_bias, data.uv_bias, data.equirect_scale);
      mul_v2_v2(data.equirect_scale, data.uv_scale);

      copy_v2_v2(data.equirect_scale_inv, data.equirect_scale);
      invert_v2(data.equirect_scale_inv);
    }
    else {
      data.clip_near = DRW_view_near_distance_get(drw_view);
      data.clip_far = DRW_view_far_distance_get(drw_view);
      data.fisheye_fov = data.fisheye_lens = -1.0f;
      copy_v2_fl(data.equirect_bias, 0.0f);
      copy_v2_fl(data.equirect_scale, 0.0f);
    }

    GPU_uniformbuf_update(current.ubo_, current.data_);

    synced_ = true;
  }

  /**
   * Getters
   **/
  const Object *blender_camera_get(void) const
  {
    BLI_assert(synced_);
    return object_eval_;
  }
  const CameraData &data_get(void) const
  {
    BLI_assert(synced_);
    return *current.data_;
  }
  const GPUUniformBuf *ubo_get(void) const
  {
    return current.ubo_;
  }
  bool has_changed(void) const
  {
    BLI_assert(synced_);
    /* TODO(fclem) This whole has_changed logic is a bit weak. To revisit. */
    return !DRW_state_is_scene_render() && has_changed_;
  }
  bool is_panoramic(void) const
  {
    return eevee::is_panoramic(current.data_->type);
  }
};

/** \} */

}  // namespace blender::eevee
