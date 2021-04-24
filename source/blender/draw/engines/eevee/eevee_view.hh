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
 * A view is either:
 * - The entire main view.
 * - A fragment of the main view (for panoramic projections).
 * - A shadow map view.
 * - A lightprobe view (either planar, cubemap, irradiance grid).
 *
 * A pass is a container for scene data. It is view agnostic but has specific logic depending on
 * its type. Passes are shared between views.
 */

#pragma once

#include "GPU_framebuffer.h"
#include "GPU_texture.h"

#include "DRW_render.h"

#include "eevee_depth_of_field.hh"
#include "eevee_motion_blur.hh"
#include "eevee_renderpasses.hh"
#include "eevee_shader.hh"
#include "eevee_velocity.hh"

namespace blender::eevee {

class Instance;

/* -------------------------------------------------------------------- */
/** \name ShadingView
 *
 * Render the scene and fill all render passes data.
 * \{ */

class ShadingView {
 private:
  Instance &inst_;

  /** Post-fx modules. */
  DepthOfField dof_;
  MotionBlur mb_;
  Velocity velocity_;

  /** Owned resources. */
  eevee::Framebuffer view_fb_;
  eevee::Framebuffer debug_fb_;
  /** Draw resources. Not owned. */
  GPUTexture *combined_tx_ = nullptr;
  GPUTexture *depth_tx_ = nullptr;
  GPUTexture *postfx_tx_ = nullptr;

  /** Main views is created from the camera (or is from the viewport). It is not jittered. */
  DRWView *main_view_ = nullptr;
  /** Sub views is jittered versions or the main views. This allows jitter updates without trashing
   * the visibility culling cache. */
  DRWView *sub_view_ = nullptr;
  /** Same as sub_view_ but has Depth Of Field jitter applied. */
  DRWView *render_view_ = nullptr;

  /** Render size of the view. Can change between scene sample eval. */
  ivec2 extent_ = {-1, -1};
  /** Static srting pointer. Used as debug name and as UUID for texture pool. */
  const char *name_;
  /** Matrix to apply to the viewmat. */
  const float (*face_matrix_)[4];

  bool is_enabled_ = false;

 public:
  ShadingView(Instance &inst, const char *name, const float (*face_matrix)[4])
      : inst_(inst),
        dof_(inst, name),
        mb_(inst, name),
        velocity_(inst, name),
        name_(name),
        face_matrix_(face_matrix){};

  ~ShadingView(){};

  void init(void);

  void sync(ivec2 render_extent_);

  void render(void);

  GPUTexture *render_post(GPUTexture *input_tx);

 private:
  void update_view(void);
};

/** \} */

/* -------------------------------------------------------------------- */
/** \name Main View
 *
 * \{ */

/**
 * Container for all views needed to render the final image.
 * We might need up to 6 views for panoramic cameras.
 */
class MainView {
 private:
  std::array<ShadingView, 6> shading_views_;
  /** Internal render size. */
  int render_extent_[2];

 public:
  MainView(Instance &inst)
      : shading_views_({
            ShadingView(inst, "posX_view", cubeface_mat[0]),
            ShadingView(inst, "negX_view", cubeface_mat[1]),
            ShadingView(inst, "posY_view", cubeface_mat[2]),
            ShadingView(inst, "negY_view", cubeface_mat[3]),
            ShadingView(inst, "posZ_view", cubeface_mat[4]),
            ShadingView(inst, "negZ_view", cubeface_mat[5]),
        })
  {
  }

  void init(const ivec2 full_extent_)
  {
    /* TODO(fclem) parameter hidden in experimental. We need to figure out mipmap bias to preserve
     * texture crispiness. */
    float resolution_scale = 1.0f;
    for (int i = 0; i < 2; i++) {
      render_extent_[i] = max_ii(1, roundf(full_extent_[i] * resolution_scale));
    }

    for (ShadingView &view : shading_views_) {
      view.init();
    }
  }

  void sync(void)
  {
    for (ShadingView &view : shading_views_) {
      view.sync(render_extent_);
    }
  }

  void render(void)
  {
    for (ShadingView &view : shading_views_) {
      view.render();
    }
  }
};

/** \} */

}  // namespace blender::eevee