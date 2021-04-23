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
 *
 * The shadow module manages shadow update tagging & shadow rendering.
 */

#pragma once

#include "BLI_vector.hh"

#include "eevee_id_map.hh"
#include "eevee_shader.hh"
#include "eevee_shader_shared.hh"

namespace blender::eevee {

class Instance;

struct ShadowRegion {
  /** Position and size in the shadow atlas. */
  int offset[2];
  int extent[2];
  /** View used for rendering the region. */
  DRWView *view = nullptr;
  /** Update tag. */
  bool do_update;

  ShadowRegion(int w, int h)
  {
    extent[0] = w;
    extent[1] = h;
  }

  ShadowRegionData to_region_data(const int atlas_extent[2])
  {
    ShadowRegionData data;
    DRW_view_winmat_get(view, data.shadow_mat, false);

    /**
     * Conversion from NDC to atlas coordinate.
     * GLSL pseudo code:
     * region_uv = ((ndc * 0.5 + 0.5) * extent + offset) / atlas_extent;
     * region_uv = ndc * (0.5 * extent) / atlas_extent
     *                 + (0.5 * extent + offset) / atlas_extent;
     * Also remove half a pixel from each side to avoid interpolation issues.
     * The projection matrix should take this into account.
     **/
    mat4 coord_mat;
    zero_m4(coord_mat);
    coord_mat[0][0] = (0.5f * (extent[0] - 1)) / atlas_extent[0];
    coord_mat[1][1] = (0.5f * (extent[1] - 1)) / atlas_extent[1];
    coord_mat[2][2] = 0.5f;
    coord_mat[3][0] = (0.5f * (extent[0] - 1) + (offset[0] + 0.5f)) / atlas_extent[0];
    coord_mat[3][1] = (0.5f * (extent[1] - 1) + (offset[1] + 0.5f)) / atlas_extent[1];
    coord_mat[3][2] = 0.5f;
    coord_mat[3][3] = 1.0f;
    mul_m4_m4m4(data.shadow_mat, coord_mat, data.shadow_mat);

    return data;
  }
};

/* -------------------------------------------------------------------- */
/** \name ShadowPunctual
 *
 * \{ */

class ShadowPunctual {
 public:
  /** Bounds to check if shadow if visible. */
  BoundSphere bsphere;
  /** Region index in the region array. -1 denote the shadow could not fit in the shadow array. */
  int region_first;
  /** Number of region used by the shadow. -1 denote uninitialized data. */
  int region_count = -1;
  /** Update tag. */
  bool do_update;
  /** Used to detect deleted objects. */
  bool alive;

 private:
  /** A wide cone has an aperture larger than 90° and covers more than 1 cubeface. */
  bool is_wide_cone_;
  /** The shadow covers the whole sphere and all faces needs to be rendered. */
  bool is_omni_;
  /** Near clip distances. Far clip distance is bsphere.radius. */
  float near_;
  /** View space offset to apply to the shadow. */
  float bias_;
  /** Shadow cube size in pixels. */
  int size_;
  /** Normalized object matrix. Used as base for view matrix creation. */
  mat4 object_mat_;

  enum eShadowCubeFace {
    /* Ordering by culling order. If cone aperture is shallow, we cull the later view. */
    Z_NEG = 0,
    X_POS,
    X_NEG,
    Y_POS,
    Y_NEG,
    Z_POS,
  };

  /* Since we don't render to cubemap textures, order or orientation is not important
   * as long as winding is left unchanged.
   * So we minimize the complexity of `half_cubeface_projmat_get()` by rotating the cropped edge
   * to always be on the top side. */
  constexpr static const float shadow_face_mat[6][4][4] = {
      {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}},   /* Z_NEG */
      {{0, 0, -1, 0}, {-1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}}, /* X_POS */
      {{0, 0, 1, 0}, {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}},   /* X_NEG */
      {{1, 0, 0, 0}, {0, 0, -1, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}},  /* Y_POS */
      {{-1, 0, 0, 0}, {0, 0, 1, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}},  /* Y_NEG */
      {{1, 0, 0, 0}, {0, -1, 0, 0}, {0, 0, -1, 0}, {0, 0, 0, 1}}, /* Z_POS */
  };

 public:
  void sync(const mat4 light_mat, float radius, float cone_aperture, float near_clip, int size)
  {
    copy_m4_m4(object_mat_, light_mat);
    /* Remove custom data packed in last column. */
    object_mat_[0][3] = object_mat_[1][3] = object_mat_[2][3] = 0.0f;
    object_mat_[3][3] = 1.0f;

    /* TODO(fclem) Better bound sphere for cones. */
    copy_v3_v3(bsphere.center, light_mat[3]);
    bsphere.radius = radius;

    is_wide_cone_ = cone_aperture > DEG2RAD(90);
    is_omni_ = cone_aperture > DEG2RAD(180);
    near_ = near_clip;
    size_ = size;
  }

  void allocate_regions(Vector<ShadowRegion> &regions)
  {
    /* -Z. */
    regions.append(ShadowRegion(size_, size_));
    if (is_omni_) {
      /* +X, -X, +Y, -Y, +Z. */
      regions.append_n_times(ShadowRegion(size_, size_), 5);
    }
    else if (is_wide_cone_) {
      /* +X, -X, +Y, -Y. */
      regions.append_n_times(ShadowRegion(size_, size_ / 2), 4);
    }
  }

  void prepare_views(MutableSpan<ShadowRegion> regions)
  {
    float jittered_mat[4][4];
    float winmat[4][4];

    /* TODO(fclem) jitter. */
    copy_m4_m4(jittered_mat, object_mat_);

    /* The jittered_mat is garanteed to be normalized by Light::Light(), so transpose is also the
     * inverse. Saves some processing. Transpose is equivalent to inverse only if matrix is
     * symmetrical. */
    /* TODO(fclem) */
    // transpose_m4(r_viewmat);
    /* Apply translation. */
    // translate_m4(r_viewmat, );
    invert_m4(jittered_mat);

    cubeface_projmat_get(winmat, false);

    region_update(regions[Z_NEG], jittered_mat, winmat, Z_NEG);

    if (is_wide_cone_) {
      if (!is_omni_) {
        cubeface_projmat_get(winmat, true);
      }
      region_update(regions[X_POS], jittered_mat, winmat, X_POS);
      region_update(regions[X_NEG], jittered_mat, winmat, X_NEG);
      region_update(regions[Y_POS], jittered_mat, winmat, Y_POS);
      region_update(regions[Y_NEG], jittered_mat, winmat, Y_NEG);
    }
    if (is_omni_) {
      region_update(regions[Z_POS], jittered_mat, winmat, Z_POS);
    }
  }

 private:
  void cubeface_projmat_get(mat4 winmat, bool half_opened)
  {
    /* Open the frustum a bit more to align border pixels with the different views. */
    float side = near_ * size_ / float(size_ - 1);
    float far = bsphere.radius;
    perspective_m4(winmat, -side, side, -side, (half_opened) ? 0.0f : side, near_, far);
  }

  void region_update(ShadowRegion &region,
                     const mat4 jittered_mat,
                     const mat4 winmat,
                     eShadowCubeFace face)
  {
    mat4 facemat;
    mul_m4_m4m4(facemat, winmat, shadow_face_mat[face]);

    if (region.view == nullptr) {
      region.view = DRW_view_create(jittered_mat, facemat, nullptr, nullptr, nullptr);
    }
    else {
      DRW_view_update(region.view, jittered_mat, facemat, nullptr, nullptr);
    }
  }
};

/** \} */

/* -------------------------------------------------------------------- */
/** \name ShadowModule
 *
 * Manages shadow atlas and shadow region datas.
 * \{ */

class ShadowModule {
 private:
  Instance &inst_;

  /* All allocated regions. Can grow higher than SHADOW_REGION_MAX. */
  Vector<ShadowRegion> regions_;

  Map<ObjectKey, ShadowPunctual> point_shadows_;

  ShadowRegionDataBuf regions_data_;

  /**
   * For now we store every shadow in one atlas for simplicity.
   * We could split by light batch later to improve vram usage.
   */
  eevee::Texture atlas_tx_ = Texture("shadow_atlas_tx");
  eevee::Framebuffer atlas_fb_ = Framebuffer("shadow_fb");

  bool soft_shadows_enabled_ = false;

  GPUTexture *atlas_tx_ptr_;

 public:
  ShadowModule(Instance &inst) : inst_(inst){};
  ~ShadowModule(){};

  void begin_sync(void);
  int sync_punctual_shadow(const ObjectHandle &ob_handle,
                           const mat4 light_mat,
                           float radius,
                           float cone_aperture,
                           float near_clip);
  void end_sync(void);

  void set_view(const DRWView *view);

  const GPUUniformBuf *regions_ubo_get(void)
  {
    return regions_data_.ubo_get();
  }
  GPUTexture **atlas_ref_get(void)
  {
    return &atlas_tx_ptr_;
  }
};

/** \} */

/* -------------------------------------------------------------------- */
/** \name ShadowPass
 *
 * A simple depth pass to which all shadow casters subscribe.
 * \{ */

class ShadowPass {
 private:
  Instance &inst_;

  DRWPass *clear_ps_ = nullptr;
  DRWPass *surface_ps_ = nullptr;

  /** Shading groups from surface_ps_ */
  DRWShadingGroup *surface_grp_;

 public:
  ShadowPass(Instance &inst) : inst_(inst){};

  void sync(void);

  void surface_add(Object *ob, Material *mat, int matslot);

  void render(void);
};

/** \} */

}  // namespace blender::eevee
