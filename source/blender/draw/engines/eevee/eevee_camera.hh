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

#include "DRW_render.h"

#include "BKE_camera.h"

#include "RE_pipeline.h"

#include "DNA_camera_types.h"
#include "DNA_object_types.h"
#include "DNA_view3d_types.h"

#include "eevee_sampling.hh"
#include "eevee_shader_shared.hh"

namespace blender::eevee {

/* Cube-map Matrices */
static const float cubeface_matrix[6][4][4] = {
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

typedef struct Camera {
 private:
  eCameraType type_;
  /** Main views are created from the camera (or is from the viewport). They are not jittered. */
  Vector<DRWView *, 6> main_views_;
  /** Sub views are jittered versions or the main views. */
  Vector<DRWView *, 6> sub_views_;
  /** Random module to know what jitter to apply to the view. */
  Sampling &sampling_;
  /** Double buffered to detect changes and have history for re-projection. */
  struct {
    CameraData *data_;
    GPUUniformBuf *ubo_;
  } current, previous;
  /** True if camera matrix has change since last init. */
  bool has_changed_ = true;
  /** Reset by init(). Makes sure only first sync detects changes. */
  bool is_init_ = false;

 public:
  Camera(Sampling &sampling) : sampling_(sampling)
  {
    current.data_ = (CameraData *)MEM_callocN(sizeof(CameraData), "CameraData");
    current.ubo_ = GPU_uniformbuf_create_ex(sizeof(CameraData), nullptr, "CameraData");

    previous.data_ = (CameraData *)MEM_callocN(sizeof(CameraData), "CameraData");
    previous.ubo_ = GPU_uniformbuf_create_ex(sizeof(CameraData), nullptr, "CameraData");

    /* Alloc at least 6 view for panoramic projections. */
    main_views_.resize(6);
    sub_views_.resize(6);
  };

  ~Camera()
  {
    MEM_SAFE_FREE(current.data_);
    MEM_SAFE_FREE(previous.data_);
    DRW_UBO_FREE_SAFE(current.ubo_);
    DRW_UBO_FREE_SAFE(previous.ubo_);
  };

  void init(const Object *camera_object_eval, const DRWView *drw_view)
  {
    SWAP(CameraData *, current.data_, previous.data_);
    SWAP(GPUUniformBuf *, current.ubo_, previous.ubo_);

    CameraData &data = *current.data_;

    if (drw_view) {
      DRW_view_camtexco_get(drw_view, &data.uv_scale[0]);
    }
    else {
      copy_v2_fl(data.uv_scale, 1.0f);
      copy_v2_fl(data.uv_bias, 0.0f);
    }
    /* These settings need to be set early and are immutable for the entire frame.
     * This means no motion blur animation support for these. */
    if (camera_object_eval) {
      const ::Camera *cam = reinterpret_cast<const ::Camera *>(camera_object_eval->data);
      data.type = from_camera(cam);
      data.near_clip = cam->clip_start;
      data.far_clip = cam->clip_end;
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
      data.type = DRW_view_is_persp_get(drw_view) ? CAMERA_PERSP : CAMERA_ORTHO;
      data.near_clip = DRW_view_near_distance_get(drw_view);
      data.far_clip = DRW_view_far_distance_get(drw_view);
      data.fisheye_fov = data.fisheye_lens = -1.0f;
      copy_v2_fl(data.equirect_bias, 0.0f);
      copy_v2_fl(data.equirect_scale, 0.0f);
    }

    is_init_ = false;
  }

  void sync(const RenderEngine *engine,
            const Object *camera_object_eval,
            const DRWView *drw_view,
            float filter_size,
            float overscan)
  {
    CameraData &data = *current.data_;

    data.filter_size = filter_size;

    if (drw_view) {
      DRW_view_viewmat_get(drw_view, data.viewmat, false);
      DRW_view_viewmat_get(drw_view, data.viewinv, true);
      DRW_view_winmat_get(drw_view, data.winmat, false);
      DRW_view_winmat_get(drw_view, data.wininv, true);
      DRW_view_persmat_get(drw_view, data.persmat, false);
      DRW_view_persmat_get(drw_view, data.persinv, true);
    }
    else {
      /* TODO(fclem) Overscan */
      (void)overscan;
      // RE_GetCameraWindowWithOverscan(engine->re, g_data->overscan, data.winmat);
      RE_GetCameraWindow(engine->re, camera_object_eval, data.winmat);
      RE_GetCameraModelMatrix(engine->re, camera_object_eval, data.viewinv);
      invert_m4_m4(data.viewmat, data.viewinv);
      invert_m4_m4(data.wininv, data.winmat);
      mul_m4_m4m4(data.persmat, data.winmat, data.viewmat);
      invert_m4_m4(data.persinv, data.persmat);
    }

    memset(main_views_.data(), 0, sizeof(main_views_[0]) * main_views_.size());
    memset(sub_views_.data(), 0, sizeof(sub_views_[0]) * sub_views_.size());

    if (this->is_panoramic()) {
      float winmat[4][4], near = data.near_clip, far = data.far_clip;
      /* TODO(fclem) Overscans. */
      perspective_m4(winmat, -near, near, -near, near, near, far);

      for (int i = view_count_get() - 1; i >= 0; i--) {
        float viewmat[4][4];
        mul_m4_m4m4(viewmat, cubeface_matrix[i], data.viewmat);

        if (main_views_[i] == nullptr) {
          main_views_[i] = DRW_view_create(viewmat, winmat, nullptr, nullptr, nullptr);
          sub_views_[i] = DRW_view_create_sub(main_views_[i], viewmat, winmat);
        }
        else {
          DRW_view_update(main_views_[i], viewmat, winmat, nullptr, nullptr);
          DRW_view_update_sub(sub_views_[i], viewmat, winmat);
        }
      }
    }
    else {
      if (main_views_[0] == nullptr) {
        main_views_[0] = DRW_view_create(data.viewmat, data.winmat, nullptr, nullptr, nullptr);
        sub_views_[0] = DRW_view_create_sub(main_views_[0], data.viewmat, data.winmat);
      }
      else {
        DRW_view_update(main_views_[0], data.viewmat, data.winmat, nullptr, nullptr);
        DRW_view_update_sub(main_views_[0], data.viewmat, data.winmat);
      }
    }

    /* Detect changes in parameters. */
    if (!is_init_) {
      is_init_ = true;
      has_changed_ = *current.data_ != *previous.data_;
      if (has_changed_) {
        sampling_.reset();
      }
    }
  }

  void end_sync(void)
  {
    GPU_uniformbuf_update(current.ubo_, current.data_);
  }

  /* Apply jittering to the view and returns it. */
  DRWView *update_view(int view_id, int target_res[2])
  {
    const DRWView *main_view = main_views_[view_id];
    DRWView *sub_view = sub_views_[view_id];

    float viewmat[4][4], winmat[4][4], persmat[4][4];
    DRW_view_viewmat_get(main_view, viewmat, false);
    DRW_view_winmat_get(main_view, winmat, false);
    DRW_view_persmat_get(main_view, persmat, false);

    /* Apply jitter. */
    float jitter[2];
    sampling_.camera_lds_get(jitter);
    for (int i = 0; i < 2; i++) {
      jitter[i] = 2.0f * (jitter[i] - 0.5f) / target_res[i];
    }

    window_translate_m4(winmat, persmat, UNPACK2(jitter));

    DRW_view_update_sub(sub_view, viewmat, winmat);

    return sub_view;
  }

  const CameraData &data_get(void)
  {
    return *current.data_;
  }
  const GPUUniformBuf *ubo_get(void)
  {
    return current.ubo_;
  }

  int view_count_get(void) const
  {
    /* TODO(fclem) we might want to clip unused views in panoramic projections. */
    return this->is_panoramic() ? 6 : 1;
  }

  bool has_changed(void) const
  {
    return has_changed_;
  }

  bool is_panoramic(void) const
  {
    return !ELEM(current.data_->type, CAMERA_PERSP, CAMERA_ORTHO);
  }
} Camera;

/** \} */

}  // namespace blender::eevee
