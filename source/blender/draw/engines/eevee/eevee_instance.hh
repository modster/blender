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
 * An instance contains all structures needed to do a complete render.
 */

#pragma once

#include "BLI_vector.hh"

#include "eevee_film.hh"
#include "eevee_renderpasses.hh"
#include "eevee_sampling.hh"
#include "eevee_shaders.hh"
#include "eevee_view.hh"

namespace blender::eevee {

typedef struct Instance {
 public:
  /** Outputs passes. */
  RenderPasses render_passes;
  /** Shading passes. Shared between views. Objects will subscribe to one of them. */
  ShadingPasses shading_passes;
  /** Shader module. shared between instances. */
  ShaderModule &shaders;
  /** Lookdev own lightweight instance. May not be allocated. */
  // Lookdev *lookdev = nullptr;

 private:
  /** Random number generator, this is its persistent state. */
  Sampling sampling_;
  /** Shaded view for the main output. */
  Vector<ShadingView> shading_views_;
  /** Point of view in the scene. Can be init from viewport. */
  Camera camera_;

  Scene *scene_ = nullptr;
  ViewLayer *view_layer_ = nullptr;
  Depsgraph *depsgraph_ = nullptr;
  /** Main view if used in a viewport without a camera object. */
  const DRWView *viewport_drw_view_ = nullptr;

 public:
  Instance(ShaderModule &shared_shaders)
      : render_passes(shared_shaders, camera_), shaders(shared_shaders), camera_(sampling_){};
  ~Instance(){};

  /* Init funcion that needs to be called once at the start of a frame.
   * Active camera, render extent and enabled render passes are immutable until next init.
   * This takes care of resizing output buffers and view in case a parameter changed. */
  void init(const int output_res[2],
            Scene *scene,
            ViewLayer *view_layer,
            Depsgraph *depsgraph,
            Object *camera_object = nullptr,
            const DRWView *drw_view = nullptr,
            const RegionView3D *rv3d = nullptr)
  {
    BLI_assert(camera_object || drw_view);

    scene_ = scene;
    view_layer_ = view_layer;
    depsgraph_ = depsgraph;
    viewport_drw_view_ = drw_view;

    sampling_.init(scene);
    camera_.init(camera_object, drw_view, rv3d, scene->r.gauss);

    eRenderPassBit render_passes_bits = RENDERPASS_COMBINED | RENDERPASS_DEPTH;
    render_passes.configure(render_passes_bits, output_res);

    /* Init internal render view(s). */
    float resolution_scale = 1.0f; /* TODO(fclem) parameter. */
    int render_res[2];
    for (int i = 0; i < 2; i++) {
      render_res[i] = max_ii(1, roundf(output_res[i] * resolution_scale));
    }

    int view_count = camera_.view_count_get();
    if (shading_views_.size() != view_count) {
      /* FIXME(fclem) Strange, seems like resizing half-clears the objects? */
      shading_views_.clear();
      shading_views_.resize(view_count);
    }

    if (camera_.is_panoramic()) {
      int64_t render_pixel_count = render_res[0] * (int64_t)render_res[0];
      /* Divide pixel count between the 6 views. Rendering to a square target. */
      render_res[0] = render_res[1] = ceilf(sqrtf(1 + (render_pixel_count / 6)));

      static const char *view_names[6] = {
          "posX_view", "negX_view", "posY_view", "negY_view", "posZ_view", "negZ_view"};
      for (int i = 0; i < view_count; i++) {
        shading_views_[i].configure(
            view_names[i], render_passes, shading_passes, sampling_, camera_, i, render_res);
      }
    }
    else {
      shading_views_[0].configure(
          "main_view", render_passes, shading_passes, sampling_, camera_, 0, render_res);
    }
  }

  void begin_sync(void)
  {
    render_passes.init();
    shading_passes.init(shaders);

    for (ShadingView &view : shading_views_) {
      view.init();
    }
  }

  void camera_sync(void)
  {
  }

  void object_sync(Object *ob)
  {
    switch (ob->type) {
      case OB_MESH:
        shading_passes.opaque.surface_add(ob, nullptr, 0);
        break;

      default:
        break;
    }
  }

  void end_sync(void)
  {
    camera_.end_sync();
  }

  void render_sample()
  {
    if (sampling_.finished()) {
      return;
    }

    /* TODO update shadowmaps, planars, etc... */

    for (ShadingView &view : shading_views_) {
      view.render();
    }

    sampling_.step();
  }

  bool finished(void) const
  {
    return sampling_.finished();
  }

} Instance;

}  // namespace blender::eevee
