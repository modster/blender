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

#pragma once

#include "DRW_render.h"

#include "eevee_shader_shared.hh"
#include "eevee_shading.hh"

namespace blender::eevee {

class Instance;

/* -------------------------------------------------------------------- */
/** \name Raytracing
 * \{ */

class RaytracingModule {
 private:
  Instance &inst_;

  RaytraceDataBuf reflection_data_;
  RaytraceDataBuf refraction_data_;

  bool enabled_ = false;

 public:
  RaytracingModule(Instance &inst) : inst_(inst){};

  void sync(void);

  const GPUUniformBuf *reflection_ubo_get(void) const
  {
    return reflection_data_.ubo_get();
  }
  const GPUUniformBuf *refraction_ubo_get(void) const
  {
    return refraction_data_.ubo_get();
  }

  bool enabled(void) const
  {
    return enabled_;
  }

 private:
  void generate_sample_reuse_table(void);
};

/** \} */

}  // namespace blender::eevee
