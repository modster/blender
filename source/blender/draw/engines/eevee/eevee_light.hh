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
 * The light module manages light data buffers and light culling system.
 */

#pragma once

#include "BLI_bitmap.h"
#include "BLI_vector.hh"
#include "DNA_light_types.h"

#include "eevee_camera.hh"
#include "eevee_culling.hh"
#include "eevee_id_map.hh"
#include "eevee_sampling.hh"
#include "eevee_shader.hh"
#include "eevee_shader_shared.hh"
#include "eevee_shadow.hh"
#include "eevee_wrapper.hh"

namespace blender::eevee {

class Instance;

/* -------------------------------------------------------------------- */
/** \name Light Object
 * \{ */

class Light : public LightData {
 public:
  Light(const Object *ob,
        const ObjectHandle &object_handle,
        float threshold,
        ShadowModule &shadows);

  void debug_draw(void);

 private:
  float attenuation_radius_get(const ::Light *la, float light_threshold, float light_power);
  void shape_parameters_set(const ::Light *la, const float scale[3]);
  float shape_power_get(const ::Light *la);
  float shape_power_volume_get(const ::Light *la);
};

/** \} */

/* -------------------------------------------------------------------- */
/** \name CullingPass
 * \{ */

class CullingLightPass {
 private:
  Instance &inst_;

  DRWPass *culling_ps_ = nullptr;
  const GPUUniformBuf *lights_ubo_ = nullptr;
  const GPUUniformBuf *culling_ubo_ = nullptr;

 public:
  CullingLightPass(Instance &inst) : inst_(inst){};

  void sync(void);
  void render(const GPUUniformBuf *lights_ubo, const GPUUniformBuf *culling_ubo);
};

/** \} */

/* -------------------------------------------------------------------- */
/** \name LightModule
 * \{ */

/**
 * The light module manages light data buffers and light culling system.
 */
class LightModule {
 private:
  Instance &inst_;

  /** Map of light objects. This is used to track light deletion. */
  Map<ObjectKey, bool> objects_light_;
  /** Gathered Light data from sync. Not all data will be selected for rendering. */
  Vector<Light> lights_;
  /** Batches of lights alongside their culling data. */
  Culling<Light, LightData, CullingLightPass, true> culling_;
  /** Active data pointers used for rendering. */
  const GPUUniformBuf *active_data_ubo_;
  const GPUUniformBuf *active_culling_ubo_;
  GPUTexture *active_culling_tx_;

  uint64_t active_batch_count_;

  float light_threshold_;

 public:
  LightModule(Instance &inst) : inst_(inst), culling_(lights_){};
  ~LightModule(){};

  void begin_sync(void);
  void sync_light(const Object *ob, ObjectHandle &handle);
  void end_sync(void);

  void set_view(const DRWView *view, const ivec2 extent);

  void bind_batch(int range_id);

  /**
   * Getters
   **/
  const GPUUniformBuf **data_ubo_ref_get(void)
  {
    return &active_data_ubo_;
  }
  const GPUUniformBuf **culling_ubo_ref_get(void)
  {
    return &active_culling_ubo_;
  }
  GPUTexture **culling_tx_ref_get(void)
  {
    return &active_culling_tx_;
  }
  /* Return a range iterator to loop over all lights.
   * In practice, we render with light in waves of LIGHT_MAX lights at a time. */
  IndexRange index_range(void) const
  {
    return culling_.index_range();
  }
};

/** \} */

}  // namespace blender::eevee
