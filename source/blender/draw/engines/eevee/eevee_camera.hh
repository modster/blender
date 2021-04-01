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

#include "DNA_camera_types.h"
#include "DNA_object_types.h"
#include "DNA_view3d_types.h"

#include "eevee_sampling.hh"
#include "eevee_shared.hh"

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
  Vector<const DRWView *> main_views_;
  /** Sub views are jittered versions or the main views. */
  Vector<DRWView *> sub_views_;
  /** Random module to know what jitter to apply to the view. */
  Sampling &sampling_;
  /** Double buffered to detect changes and have history for re-projection. */
  struct {
    CameraData *data_;
    GPUUniformBuf *ubo_;
  } current, previous;
  /** True if camera matrix has change since last init. */
  bool has_changed_;

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

  void init(const Object *camera_object,
            const DRWView *drw_view,
            const RegionView3D *rv3d,
            float filter_size)
  {
    SWAP(CameraData *, current.data_, previous.data_);
    SWAP(GPUUniformBuf *, current.ubo_, previous.ubo_);

    CameraData &data = *current.data_;

    this->sync(camera_object, drw_view, rv3d, filter_size);

    has_changed_ = data != *previous.data_;

    if (has_changed_) {
      sampling_.reset();
    }

    int view_count = view_count_get();
    if (main_views_.size() != view_count) {
      main_views_.resize(view_count);
      sub_views_.resize(view_count);
    }

    if (this->is_panoramic()) {
      float winmat[4][4], near = data.near_clip, far = data.far_clip;
      /* TODO(fclem) Overscans. */
      perspective_m4(winmat, -near, near, -near, near, near, far);

      for (int i = 0; i < view_count; i++) {
        float viewmat[4][4];
        mul_m4_m4m4(viewmat, cubeface_matrix[i], data.viewmat);

        main_views_[i] = DRW_view_create(viewmat, winmat, nullptr, nullptr, nullptr);
        sub_views_[i] = DRW_view_create_sub(main_views_[i], viewmat, winmat);
      }
    }
    else {
      float(*winmat)[4] = data.winmat;
      float(*viewmat)[4] = data.viewmat;
      main_views_[0] = DRW_view_create(viewmat, winmat, nullptr, nullptr, nullptr);
      sub_views_[0] = DRW_view_create_sub(main_views_[0], viewmat, winmat);
    }
  }

  void sync(const Object *camera_object,
            const DRWView *drw_view,
            const RegionView3D *rv3d,
            float filter_size)
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
      /* TODO(fclem) Render */
    }

    if (camera_object) {
      if (rv3d) {
        copy_v2_v2(data.uv_scale, &rv3d->viewcamtexcofac[0]);
        copy_v2_v2(data.uv_bias, &rv3d->viewcamtexcofac[2]);
      }
      else {
        copy_v2_fl(data.uv_scale, 1.0f);
        copy_v2_fl(data.uv_bias, 0.0f);
      }
      const ::Camera *cam = reinterpret_cast<const ::Camera *>(camera_object->data);
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
      copy_v2_fl(data.uv_scale, 1.0f);
      copy_v2_fl(data.uv_bias, 0.0f);
    }
  }

  void end_sync(void)
  {
    GPU_uniformbuf_update(current.ubo_, current.data_);
  }

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
