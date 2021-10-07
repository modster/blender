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
 */

#pragma once

#include "eevee_shader.hh"
#include "eevee_shader_shared.hh"
#include "eevee_wrapper.hh"

namespace blender::eevee {

/* -------------------------------------------------------------------- */
/** \name Subsurface
 *
 * \{ */

class Instance;

struct SubsurfaceModule {
 private:
  Instance &inst_;
  /** Contains samples locations. */
  SubsurfaceDataBuf data_;
  /** Contains samples weights for a single color channel. */
  Texture *weights_tx = nullptr;
  /** Contains translucence profile for a single color channel. */
  Texture *translucency_tx = nullptr;

  /** Number of substeps use by eval_integral(). */
  static constexpr int integral_resolution = 32;
  static constexpr int cdf_resolution = 128;
  /** Value of x after which we consider burley profile to be 0. */
  static constexpr float burley_truncate = 16.0f;

 public:
  SubsurfaceModule(Instance &inst) : inst_(inst)
  {
    /* Force first update. */
    data_.sample_len = -1;
  };

  ~SubsurfaceModule()
  {
    delete weights_tx;
    delete translucency_tx;
  };

  void end_sync();

  const GPUUniformBuf *ubo_get(void)
  {
    return data_.ubo_get();
  }

 private:
  void precompute_samples_location();
  void precompute_diffusion_cdf();
  void precompute_transmittance_profile();

  /** Christensen-Burley implementation. */
  static float burley_setup(float radius, float albedo);
  static float burley_sample(float d, float x_rand);
  static float burley_eval(float d, float r);
  static float burley_pdf(float d, float r);
};

/** \} */

}  // namespace blender::eevee
