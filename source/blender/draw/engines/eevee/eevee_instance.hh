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

#include "DEG_depsgraph_query.h"

#include "eevee_film.hh"
#include "eevee_renderpasses.hh"
#include "eevee_sampling.hh"
#include "eevee_shader.hh"
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
  /** Only available when rendering for final render. */
  const RenderLayer *render_layer_ = nullptr;
  /** Only available when rendering for viewport. */
  const DRWView *drw_view_ = nullptr;
  const View3D *v3d_ = nullptr;
  const RegionView3D *rv3d_ = nullptr;
  /** Original object of the camera. */
  Object *camera_original_ = nullptr;

 public:
  Instance(ShaderModule &shared_shaders)
      : render_passes(shared_shaders, camera_, sampling_),
        shaders(shared_shaders),
        camera_(sampling_){};
  ~Instance(){};

  /* Init funcion that needs to be called once at the start of a frame.
   * Active camera, render extent and enabled render passes are immutable until next init.
   * This takes care of resizing output buffers and view in case a parameter changed. */
  void init(const int output_res[2],
            const rcti *output_rect,
            Depsgraph *depsgraph,
            Object *camera_object = nullptr,
            const RenderLayer *render_layer = nullptr,
            const DRWView *drw_view = nullptr,
            const View3D *v3d = nullptr,
            const RegionView3D *rv3d = nullptr)
  {
    BLI_assert(camera_object || drw_view);

    scene_ = DEG_get_evaluated_scene(depsgraph);
    view_layer_ = DEG_get_evaluated_view_layer(depsgraph);
    depsgraph_ = depsgraph;
    camera_original_ = camera_object;
    render_layer_ = render_layer;
    drw_view_ = drw_view;
    v3d_ = v3d;
    rv3d_ = rv3d;

    rcti rect;
    {
      rcti rect_full;
      BLI_rcti_init(&rect_full, 0, output_res[0], 0, output_res[1]);
      /* Clip the render border to region bounds. */
      BLI_rcti_isect(output_rect, &rect_full, &rect);
      if (BLI_rcti_is_empty(&rect)) {
        BLI_rcti_init(&rect, 0, output_res[0], 0, output_res[1]);
      }
      output_rect = &rect;
    }

    const Object *camera_eval = DEG_get_evaluated_object(depsgraph_, camera_original_);

    sampling_.init(scene_);
    camera_.init(camera_eval, drw_view_);
    render_passes.init(scene_, render_layer, v3d_, output_res, output_rect);

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
        shading_views_[i].init(
            view_names[i], render_passes, shading_passes, sampling_, camera_, i, render_res);
      }
    }
    else {
      shading_views_[0].init(
          "main_view", render_passes, shading_passes, sampling_, camera_, 0, render_res);
    }
  }

  void begin_sync(RenderEngine *render)
  {
    const Object *camera_eval = DEG_get_evaluated_object(depsgraph_, camera_original_);

    camera_.sync(render, camera_eval, drw_view_, scene_->r.gauss, scene_->eevee.overscan);
    render_passes.sync();
    shading_passes.sync(shaders);
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

  /* Wrapper to use with DRW_render_object_iter. */
  static void object_sync(void *instance_, Object *ob, RenderEngine *engine, Depsgraph *depsgraph)
  {
    UNUSED_VARS(engine, depsgraph);
    reinterpret_cast<Instance *>(instance_)->object_sync(ob);
  }

  void end_sync(void)
  {
    camera_.end_sync();
  }

  void render_sample(void)
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

  void render_frame(RenderEngine *engine, RenderLayer *render_layer, const char *view_name)
  {
    this->begin_sync(engine);
    DRW_render_object_iter(this, engine, depsgraph_, object_sync);
    this->end_sync();

    DRW_render_instance_buffer_finish();

    /* Also we weed to have a correct fbo bound for DRW_hair_update */
    // GPU_framebuffer_bind();
    // DRW_hair_update();

    while (!sampling_.finished()) {
      this->render_sample();
      /* TODO(fclem) print progression. */
    }

    this->render_passes.read_result(render_layer, view_name);
  }

  void draw_viewport(DefaultFramebufferList *dfbl)
  {
    this->render_sample();

    this->render_passes.resolve_viewport(dfbl);

    // if (this->lookdev) {
    // this->lookdev->resolve_onto(dfbl->default_fb);
    // }

    if (!sampling_.finished()) {
      DRW_viewport_request_redraw();
    }
  }

  bool finished(void) const
  {
    return sampling_.finished();
  }

} Instance;

}  // namespace blender::eevee
