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
#include "eevee_sampling.hh"
#include "eevee_shader.hh"
#include "eevee_shader_shared.hh"
#include "eevee_wrapper.hh"

namespace blender::eevee {

class Instance;

/* -------------------------------------------------------------------- */
/** \name Light Object
 * \{ */

class Light : public LightData {
 public:
  Light(const Object *ob, float threshold);

 private:
  float inverse_squared_attenuation_radius_get(const ::Light *la,
                                               float light_threshold,
                                               float light_power);
  void shape_parameters_set(const ::Light *la, const float scale[3]);
  float shape_power_get(const ::Light *la);
  float shape_power_volume_get(const ::Light *la);
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

  /** Gathered Light data from sync. Not all data will be selected for rendering. */
  Vector<Light> lights_;

  LightsDataBuf lights_data_;
  ClustersDataBuf clusters_data_;

  float light_threshold_;

 public:
  LightModule(Instance &inst) : inst_(inst){};
  ~LightModule(){};

  void begin_sync(void);
  void sync_light(const Object *ob);
  void end_sync(void);

  void bind_range(int range_id);

  const GPUUniformBuf *ubo_get(void) const
  {
    return lights_data_.ubo_get();
  }
  const GPUUniformBuf *cluster_ubo_get(void) const
  {
    return clusters_data_.ubo_get();
  }
  /* Return a range iterator to loop over all lights.
   * In practice, we render with light in waves of LIGHT_MAX lights at a time. */
  IndexRange index_range(void) const
  {
    return IndexRange(divide_ceil_u(max_ii(1, this->lights_.size()), LIGHT_MAX));
  }
};

/** \} */

}  // namespace blender::eevee
