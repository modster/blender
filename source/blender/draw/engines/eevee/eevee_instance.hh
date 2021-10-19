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

#include "BKE_object.h"
#include "DEG_depsgraph.h"
#include "DRW_render.h"

#include "eevee_film.hh"
#include "eevee_hizbuffer.hh"
#include "eevee_id_map.hh"
#include "eevee_light.hh"
#include "eevee_lightprobe.hh"
#include "eevee_lookdev.hh"
#include "eevee_material.hh"
#include "eevee_motion_blur.hh"
#include "eevee_renderpasses.hh"
#include "eevee_sampling.hh"
#include "eevee_shader.hh"
#include "eevee_shading.hh"
#include "eevee_shadow.hh"
#include "eevee_subsurface.hh"
#include "eevee_view.hh"
#include "eevee_world.hh"

#include "eevee_engine.h"

namespace blender::eevee {

/**
 * \class Instance
 * \brief A running instance of the engine.
 */
class Instance {
  friend MotionBlur;
  friend MotionBlurModule;
  friend VelocityModule;

 public:
  ShaderModule &shaders;
  Sampling sampling;
  RenderPasses render_passes;
  ShadingPasses shading_passes;
  MainView main_view;
  Camera camera;
  World world;
  VelocityModule velocity;
  MotionBlurModule motion_blur;
  LightModule lights;
  LightProbeModule lightprobes;
  HiZBufferModule hiz;
  /* TODO(fclem) Move it to scene layer data. */
  ShadowModule shadows;
  SubsurfaceModule subsurface;
  SyncModule sync;
  MaterialModule materials;
  /** Lookdev own lightweight instance. May not be allocated. */
  LookDev lookdev;

  /** Input data. */
  Depsgraph *depsgraph;
  /** Evaluated IDs. */
  Scene *scene;
  ViewLayer *view_layer;
  Object *camera_eval_object;
  Object *camera_orig_object;
  /** Only available when rendering for final render. */
  const RenderLayer *render_layer;
  RenderEngine *render;
  /** Only available when rendering for viewport. */
  const DRWView *drw_view;
  const View3D *v3d;
  const RegionView3D *rv3d;

  /** Can be null. Used to exclude objects during baking. */
  const struct LightProbe *baking_probe = nullptr;

  /* Info string displayed at the top of the render / viewport. */
  char info[64];

 public:
  Instance(ShaderModule &shared_shaders)
      : shaders(shared_shaders),
        render_passes(*this),
        shading_passes(*this),
        main_view(*this),
        camera(*this),
        world(*this),
        velocity(*this),
        motion_blur(*this),
        lights(*this),
        lightprobes(*this),
        hiz(*this),
        shadows(*this),
        subsurface(*this),
        sync(*this),
        materials(*this),
        lookdev(*this){};
  ~Instance(){};

  void init(const ivec2 &output_res,
            const rcti *output_rect,
            RenderEngine *render,
            Depsgraph *depsgraph,
            const struct LightProbe *light_probe_ = nullptr,
            Object *camera_object = nullptr,
            const RenderLayer *render_layer = nullptr,
            const DRWView *drw_view = nullptr,
            const View3D *v3d = nullptr,
            const RegionView3D *rv3d = nullptr);

  void begin_sync(void);
  void object_sync(Object *ob);
  void end_sync(void);

  void render_sync(void);
  void render_frame(RenderLayer *render_layer, const char *view_name);

  void draw_viewport(DefaultFramebufferList *dfbl);

  bool finished(void) const;

  bool is_viewport(void)
  {
    return !DRW_state_is_scene_render();
  }

  bool use_scene_light(void) const
  {
    return (!v3d) ||
           ((v3d->shading.type == OB_MATERIAL) &&
            (v3d->shading.flag & V3D_SHADING_SCENE_LIGHTS)) ||
           ((v3d->shading.type == OB_RENDER) &&
            (v3d->shading.flag & V3D_SHADING_SCENE_LIGHTS_RENDER));
  }

  /* Do we light the scene using the HDRI setup in the viewport settings. */
  bool use_studio_light(void) const
  {
    return (v3d) && (((v3d->shading.type == OB_MATERIAL) &&
                      ((v3d->shading.flag & V3D_SHADING_SCENE_WORLD) == 0)) ||
                     ((v3d->shading.type == OB_RENDER) &&
                      ((v3d->shading.flag & V3D_SHADING_SCENE_WORLD_RENDER) == 0)));
  }

 private:
  void render_sample(void);
  static void object_sync_render(void *instance_,
                                 Object *ob,
                                 RenderEngine *engine,
                                 Depsgraph *depsgraph);

  void mesh_sync(Object *ob, ObjectHandle &ob_handle);
  void gpencil_sync(Object *ob, ObjectHandle &ob_handle);
  void hair_sync(Object *ob, ObjectHandle &ob_handle, ModifierData *modifier_data = nullptr);

  rcti output_crop(const int output_res[2], const rcti *crop);

  void set_time(float time);
  void update_eval_members(void);
};

}  // namespace blender::eevee
