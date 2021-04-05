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
  /* Number of samples in the first ring of jittered depth of field. */
  constexpr static uint64_t dof_web_density_ = 6;
  /* High number of sample for viewport infinite rendering. */
  constexpr static uint64_t infinite_sample_count_ = 0xFFFFFFu;

  /** 1 based current sample. */
  uint64_t sample_ = 1;
  /** Target sample count. */
  uint64_t sample_count_ = 64;
  /** Number of ring in the web pattern of the jittered Depth of Field. */
  uint64_t dof_ring_count_ = 0;
  /** Number of samples in the web pattern of the jittered Depth of Field. */
  uint64_t dof_sample_count_ = 1;
  /** Safeguard against illegal reset. */
  bool sync_ = false;

 public:
  void init(const Scene *scene)
  {
    sample_count_ = DRW_state_is_image_render() ? scene->eevee.taa_render_samples :
                                                  scene->eevee.taa_samples;

    if (sample_count_ == 0) {
      BLI_assert(!DRW_state_is_image_render());
      sample_count_ = infinite_sample_count_;
    }

    if (scene->eevee.flag & SCE_EEVEE_DOF_JITTER) {
      if (sample_count_ == infinite_sample_count_) {
        /* Special case for viewport continuous rendering. We clamp to a max sample
         * to avoid the jittered dof never converging. */
        dof_ring_count_ = 6;
      }
      else {
        dof_ring_count_ = web_ring_count_get(dof_web_density_, sample_count_);
      }
      dof_sample_count_ = web_sample_count_get(dof_web_density_, dof_ring_count_);
      /* Change total sample count to fill the web pattern entirely. */
      sample_count_ = divide_ceil_u(sample_count_, dof_sample_count_) * dof_sample_count_;
    }
    else {
      dof_ring_count_ = 0;
      dof_sample_count_ = 1;
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

  /**
   * Getters
   **/
  /* Returns current, 1 based, scene sample index. */
  uint64_t sample_get(void) const
  {
    return sample_;
  }
  /* Returns sample count inside the jittered depth of field web pattern. */
  uint64_t dof_ring_count_get(void) const
  {
    return dof_ring_count_;
  }
  /* Returns sample count inside the jittered depth of field web pattern. */
  uint64_t dof_sample_count_get(void) const
  {
    return dof_sample_count_;
  }
  /* Returns true if rendering has finished. */
  bool finished(void) const
  {
    return (sample_ > sample_count_);
  }

  void camera_aa_jitter_get(const int pixel_count[2], float r_vec[2])
  {
    /* TODO(fclem) we could use some persistent states to speedup the computation. */
    double r[2], offset[2];
    /* Using 2,3 primes as per UE4 Temporal AA presentation.
     * advances.realtimerendering.com/s2014/epic/TemporalAA.pptx (slide 14) */
    uint32_t primes[2] = {2, 3};
    BLI_halton_2d(primes, offset, sample_, r);
    /* Jitter sample position inside the pixel region. */
    r_vec[0] = 2.0f * ((float)r[0] - 0.5f) / pixel_count[0];
    r_vec[1] = 2.0f * ((float)r[1] - 0.5f) / pixel_count[1];
  }

  void dof_disk_sample_get(float *r_radius, float *r_theta)
  {
    if (dof_ring_count_ == 0) {
      *r_radius = *r_theta = 0.0f;
      return;
    }

    int s = sample_ - 1;
    int ring = 0;
    int ring_sample_count = 1;
    int ring_sample = 1;

    s = s * (dof_web_density_ - 1);
    s = s % dof_sample_count_;

    /* Choosing sample to we get faster convergence.
     * The issue here is that we cannot map a low descripency sequence to this sampling pattern
     * because the same sample could be choosen twice in relatively short intervals. */
    /* For now just use an ascending sequence with an offset. This gives us relatively quick
     * initial coverage and relatively high distance between samples. */
    /* TODO(fclem) We can try to order samples based on a LDS into a table to avoid duplicates.
     * The drawback would be some memory consumption and init time. */
    int samples_passed = 1;
    while (s >= samples_passed) {
      ring++;
      ring_sample_count = ring * dof_web_density_;
      ring_sample = s - samples_passed;
      ring_sample = (ring_sample + 1) % ring_sample_count;
      samples_passed += ring_sample_count;
    }

    *r_radius = ring / (float)dof_ring_count_;
    *r_theta = 2.0f * M_PI * ring_sample / (float)ring_sample_count;
  }

 private:
  /* Returns total sample count in a web pattern of the given size. */
  static uint64_t web_sample_count_get(uint64_t web_density, uint64_t ring_count)
  {
    return ((ring_count * ring_count + ring_count) / 2) * web_density + 1;
  }

  /* Returns lowest possible ring count that contains at least sample_count samples. */
  static uint64_t web_ring_count_get(uint64_t web_density, uint64_t sample_count)
  {
    /* Inversion of dof_jitter_sample_count_get(). */
    float x = 2.0f * (sample_count - 1.0f) / web_density;
    /* Solving polynomial. We only search positive solution. */
    float discriminant = 1.0f + 4.0f * x;
    return ceilf(0.5f * (sqrt(discriminant) - 1.0f));
  }
};

}  // namespace blender::eevee
