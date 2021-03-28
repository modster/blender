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

#include "eevee_accumulator.hh"
#include "eevee_random.hh"
#include "eevee_renderpasses.hh"
#include "eevee_shaders.hh"
#include "eevee_view.hh"

using namespace blender;

typedef struct EEVEE_Instance {
 public:
  /** Outputs passes. */
  EEVEE_RenderPasses render_passes;
  /** Shading passes. Shared between views. Objects will subscribe to one of them. */
  EEVEE_ShadingPasses shading_passes;
  /** Shader module. shared between instances. */
  EEVEE_Shaders &shaders;
  /** Lookdev own lightweight instance. May not be allocated. */
  // EEVEE_Lookdev *lookdev = nullptr;

 private:
  /** Random number generator, this is its persistent state. */
  EEVEE_Random random_;
  /** Shaded view for the main output. */
  Vector<EEVEE_ShadingView> shading_views_;
  /** Point of view in the scene. Can be init from viewport. */
  EEVEE_Camera camera_;

  Scene *scene_ = nullptr;
  ViewLayer *view_layer_ = nullptr;
  Depsgraph *depsgraph_ = nullptr;
  /** Main view if used in a viewport without a camera object. */
  const DRWView *viewport_drw_view_ = nullptr;

 public:
  EEVEE_Instance(EEVEE_Shaders &shared_shaders)
      : render_passes(shared_shaders), shaders(shared_shaders){};
  ~EEVEE_Instance(void){};

  /* Init funcion that needs to be called once at the start of a frame.
   * Active camera, render resolution and enabled render passes are set in stone after this. */
  void init(int output_res[2],
            Scene *scene,
            ViewLayer *view_layer,
            Depsgraph *depsgraph,
            Object *camera_object = nullptr,
            const DRWView *drw_view = nullptr)
  {
    BLI_assert(camera_object || drw_view);

    scene_ = scene;
    view_layer_ = view_layer;
    depsgraph_ = depsgraph;
    viewport_drw_view_ = drw_view;

    if (camera_object) {
      camera_.init(camera_object);
    }
    else if (drw_view) {
      camera_.init(drw_view);
    }

    EEVEE_AccumulatorParameters accum_params;
    accum_params.res[0] = output_res[0];
    accum_params.res[1] = output_res[1];
    accum_params.filter_size = scene->r.gauss;
    accum_params.projection = camera_.projection;

    eEEVEERenderPassBit render_passes_bits = COMBINED;
    render_passes.configure(render_passes_bits, accum_params);

    /* Init internal render view(s). */
    float resolution_scale = 1.0f; /* TODO(fclem) parameter. */
    int render_res[2];
    for (int i = 0; i < 2; i++) {
      render_res[i] = max_ii(1, roundf(output_res[i] * resolution_scale));
    }

    if (ELEM(camera_.projection, ORTHO, PERSP) || true) {
      if (shading_views_.size() != 1) {
        shading_views_.resize(1);
      }
      shading_views_[0].configure(
          "main_view", render_passes, shading_passes, random_, camera_, render_res);
    }
    else {
      /* TODO(fclem) Panoramic projection. */
    }
  }

  void begin_sync(void)
  {
    render_passes.init();
    shading_passes.init(shaders);

    for (EEVEE_ShadingView &view : shading_views_) {
      view.init();
    }
  }

  void camera_sync(void)
  {
  }

  void object_sync(Object *ob)
  {
    if (ob->type == OB_MESH) {
      shading_passes.opaque.surface_add(ob, nullptr, 0);
    }
  }

  void end_sync(void)
  {
  }

  /* Return false if accumulation has finished. */
  bool render_sample()
  {
    /* For testing */
    random_.reset();

    bool do_sample = random_.step();

    if (!do_sample) {
      return false;
    }

    /* TODO update shadowmaps, planars, etc... */

    for (EEVEE_ShadingView &view : shading_views_) {
      view.render();
    }

    return true;
  }

} EEVEE_Instance;
