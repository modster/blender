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
class ShadowModule;

struct AtlasRegion {
  ivec2 offset;
  ivec2 extent;
};

/* -------------------------------------------------------------------- */
/** \name Shadow
 *
 * \{ */

struct ShadowPunctual : public AtlasRegion {
 public:
  /** Update flag. */
  bool do_update = true;
  /** The light module tag each shadow intersecting the current view. */
  bool is_visible = false;
  /** False if the object is only allocated, waiting to be reused. */
  bool used = true;

 private:
  /** A wide cone has an aperture larger than 90° and covers more than 1 cubeface. */
  bool is_wide_cone_;
  /** The shadow covers the whole sphere and all faces needs to be rendered. */
  bool is_omni_;
  /** Clip distances. */
  float near_, far_;
  /** View space offset to apply to the shadow. */
  float bias_;
  /** Shadow size in pixels. */
  int cube_res_;
  /** Copy of normalized object matrix. Used to create DRWView. */
  mat4 object_mat_;
  /** Full atlas size. Only updated after end_sync(). */
  ShadowModule &shadows_;

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
  ShadowPunctual(ShadowModule &shadows, int cube_res) : cube_res_(cube_res), shadows_(shadows){};

  void sync(
      const mat4 &object_mat, float cone_aperture, float near_clip, float far_clip, float bias);
  void render(Instance &inst, MutableSpan<DRWView *> views);

  void update_extent(int cube_res);

  operator ShadowPunctualData();

 private:
  void cubeface_winmat_get(mat4 &winmat, bool half_opened);
  void view_update(DRWView *&view, const mat4 &viewmat, const mat4 &winmat, eShadowCubeFace face);
  void view_render(Instance &inst, DRWView *views, eShadowCubeFace face);
};

/** \} */

/* -------------------------------------------------------------------- */
/** \name ShadowModule
 *
 * Manages shadow atlas and shadow region datas.
 * \{ */

class ShadowModule {
  friend ShadowPunctual;

 private:
  Instance &inst_;

  Vector<ShadowPunctual> punctuals_;
  /** First unused item in the vector for fast reallocating. */
  int punctual_unused_first_ = INT_MAX;
  /** Unused item count in the vector for fast reallocating. */
  int punctual_unused_count_ = 0;
  /** True if a shadow was deleted or allocated and we need to repack the data. */
  bool packing_changed_ = true;
  /** True if a shadow type was changed and all shadows need update. */
  bool format_changed_ = true;
  /** Full atlas size. Only updated after end_sync(). */
  ivec2 atlas_extent_;

  /**
   * TODO(fclem) These should be stored inside the Shadow objects instead.
   * The issues is that only 32 DRWView can have effective culling data with the current
   * implementation. So we try to reduce the number of DRWView allocated to avoid the slow path.
   **/
  DRWView *views_[6] = {nullptr};

  /**
   * For now we store every shadow in one atlas for simplicity.
   * We could split by light batch later to improve vram usage.
   */
  eevee::Texture atlas_tx_ = Texture("shadow_atlas_tx");
  eevee::Framebuffer atlas_fb_ = Framebuffer("shadow_fb");

  /** Scene immutable parameter. */
  int cube_shadow_res_ = 64;
  bool soft_shadows_enabled_ = false;
  eGPUTextureFormat shadow_format_;

  GPUTexture *atlas_tx_ptr_;

 public:
  ShadowModule(Instance &inst) : inst_(inst){};
  ~ShadowModule(){};

  void init(void);
  void end_sync(void);

  void update_visible(const DRWView *view);

  int punctual_new(void)
  {
    packing_changed_ = true;
    if (punctual_unused_count_ > 0) {
      /* Reallocate unused item. */
      int index = punctual_unused_first_;

      punctual_unused_count_ -= 1;
      punctual_unused_first_ = INT_MAX;

      if (punctual_unused_count_ > 0) {
        /* Find next first unused. */
        for (auto i : IndexRange(index + 1, punctuals_.size() - index - 1)) {
          if (punctuals_[i].used == false) {
            punctual_unused_first_ = i;
            break;
          }
        }
      }
      punctuals_[index].used = true;
      return index;
    }
    punctuals_.append(ShadowPunctual(*this, cube_shadow_res_));
    return punctuals_.size() - 1;
  }

  void punctual_discard(int index)
  {
    packing_changed_ = true;
    punctual_unused_count_ += 1;
    punctual_unused_first_ = min_ii(index, punctual_unused_first_);
    punctuals_[index].used = false;
  }

  ShadowPunctual &punctual_get(int index)
  {
    return punctuals_[index];
  }

  GPUTexture **atlas_ref_get(void)
  {
    return &atlas_tx_ptr_;
  }

 private:
  void remove_unused(void);
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
