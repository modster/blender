/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2021 Blender Foundation.
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
  /** Contains translucence profile for a single color channel. */
  GPUTexture *transmittance_tx = nullptr;

 public:
  SubsurfaceModule(Instance &inst) : inst_(inst)
  {
    /* Force first update. */
    data_.sample_len = -1;
  };

  ~SubsurfaceModule()
  {
    GPU_TEXTURE_FREE_SAFE(transmittance_tx);
  };

  void end_sync();

  const GPUUniformBuf *ubo_get(void) const
  {
    return data_;
  }

  GPUTexture **transmittance_ref_get(void)
  {
    return &transmittance_tx;
  }

 private:
  void precompute_samples_location();
  void precompute_transmittance_profile();

  /** Christensen-Burley implementation. */
  static float burley_setup(float radius, float albedo);
  static float burley_sample(float d, float x_rand);
  static float burley_eval(float d, float r);
  static float burley_pdf(float d, float r);
};

/** \} */

}  // namespace blender::eevee
