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
 * Module containing passes and parameters used for raytracing.
 * NOTE: For now only screen space raytracing is supported.
 */

#include <fstream>
#include <iostream>

#include "eevee_instance.hh"

#include "eevee_raytracing.hh"

namespace blender::eevee {

/* -------------------------------------------------------------------- */
/** \name Raytracing
 *
 * \{ */

void RaytracingModule::sync(void)
{
  SceneEEVEE &sce_eevee = inst_.scene->eevee;

  reflection_data_.thickness = sce_eevee.ssr_thickness;
  reflection_data_.brightness_clamp = (sce_eevee.ssr_firefly_fac < 1e-8f) ?
                                          FLT_MAX :
                                          sce_eevee.ssr_firefly_fac;
  reflection_data_.max_roughness = sce_eevee.ssr_max_roughness + 0.01f;
  reflection_data_.quality = 1.0f - 0.95f * sce_eevee.ssr_quality;
  reflection_data_.bias = 0.1f + reflection_data_.quality * 0.6f;
  reflection_data_.pool_offset = inst_.sampling.sample_get() / 5;

  refraction_data_ = static_cast<RaytraceData>(reflection_data_);
  // refraction_data_.thickness = 1e16;
  /* TODO(fclem): Clamp option for refraction. */
  /* TODO(fclem): bias option for refraction. */
  /* TODO(fclem): bias option for refraction. */

  diffuse_data_ = static_cast<RaytraceData>(reflection_data_);
  diffuse_data_.max_roughness = 1.01f;

  reflection_data_.push_update();
  refraction_data_.push_update();
  diffuse_data_.push_update();

  enabled_ = (sce_eevee.flag & SCE_EEVEE_SSR_ENABLED) != 0;
}

/** \} */

}  // namespace blender::eevee
