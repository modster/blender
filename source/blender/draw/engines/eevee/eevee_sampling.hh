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
 * Random Number Generator
 */

#pragma once

#include "BLI_rand.h"
#include "DNA_scene_types.h"
#include "GPU_framebuffer.h"
#include "GPU_texture.h"

#include "DRW_render.h"

namespace blender::eevee {

class Sampling {
 private:
  /** 1 based current sample. */
  uint64_t sample_ = 1;
  /** Target sample count. */
  uint64_t sample_count_ = 64;
  /** Safeguard against illegal reset. */
  bool sync_ = false;

 public:
  void init(const Scene *scene)
  {
    sample_count_ = DRW_state_is_image_render() ? scene->eevee.taa_render_samples :
                                                  scene->eevee.taa_samples;

    if (sample_count_ == 0) {
      BLI_assert(!DRW_state_is_image_render());
      sample_count_ = 999999;
    }
    sync_ = false;
  }

  void sync(void)
  {
    sync_ = true;
  }

  void reset(void)
  {
    BLI_assert(!sync_ && "Attempted to reset sampling after init().");
    sample_ = 1;
  }

  void step(void)
  {
    sample_++;
  }

  uint64_t sample_get(void) const
  {
    return sample_;
  }

  void camera_lds_get(float r_vec[2])
  {
    /* TODO(fclem) we could use some persistent states to speedup the computation. */
    double r[2], offset[2];
    uint32_t primes[2] = {2, 3};
    BLI_halton_2d(primes, offset, sample_, r);
    r_vec[0] = r[0];
    r_vec[1] = r[1];
  }

  bool finished(void) const
  {
    return (sample_ > sample_count_);
  }
};

}  // namespace blender::eevee
