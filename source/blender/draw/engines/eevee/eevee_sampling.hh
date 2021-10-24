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

#include "BKE_colortools.h"
#include "BLI_rand.h"
#include "BLI_vector.hh"
#include "DNA_scene_types.h"
#include "DRW_render.h"
#include "GPU_framebuffer.h"
#include "GPU_texture.h"

#include "eevee_shader_shared.hh"
#include "eevee_wrapper.hh"

namespace blender::eevee {

/**
 * Random number generator, contains persistent state and sample count logic.
 */
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
  /** Motion blur steps. */
  uint64_t motion_blur_steps_ = 1;
  /** Used for viewport smooth transition. */
  int64_t sample_viewport_ = 1;
  int64_t viewport_smoothing_start = 0;
  int64_t viewport_smoothing_duration = 0;
  /** Tag to reset sampling for the next sample. */
  bool reset_ = false;

  StructBuffer<SamplingData> data_;

 public:
  Sampling(){};
  ~Sampling(){};

  void init(const Scene *scene)
  {
    sample_count_ = DRW_state_is_image_render() ? scene->eevee.taa_render_samples :
                                                  scene->eevee.taa_samples;

    if (sample_count_ == 0) {
      BLI_assert(!DRW_state_is_image_render());
      sample_count_ = infinite_sample_count_;
    }

    motion_blur_steps_ = DRW_state_is_image_render() ? scene->eevee.motion_blur_steps : 1;
    sample_count_ = divide_ceil_u(sample_count_, motion_blur_steps_);

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

    /* Only multiply after to have full the full DoF web pattern for each time steps. */
    sample_count_ *= motion_blur_steps_;

    if (!DRW_state_is_image_render()) {
      /* TODO(fclem) UI. */
      viewport_smoothing_start = 16;
      viewport_smoothing_duration = 32;
      /* At minima start when rendering has finished. */
      viewport_smoothing_start = min_ii(viewport_smoothing_start, sample_count_);
      /* Basically counts the number of redraw. */
      sample_viewport_ += 1;
    }
    else {
      viewport_smoothing_start = 0;
      viewport_smoothing_duration = 0;
    }
  }

  void end_sync(void)
  {
    if (reset_) {
      sample_ = 1;
      sample_viewport_ = 1;
    }
  }

  void step(void)
  {
    {
      /* TODO(fclem) we could use some persistent states to speedup the computation. */
      double r[2], offset[2] = {0, 0};
      /* Using 2,3 primes as per UE4 Temporal AA presentation.
       * advances.realtimerendering.com/s2014/epic/TemporalAA.pptx (slide 14) */
      uint32_t primes[2] = {2, 3};
      BLI_halton_2d(primes, offset, sample_, r);
      data_.dimensions[SAMPLING_FILTER_U][0] = r[0];
      data_.dimensions[SAMPLING_FILTER_V][0] = r[1];
      /* TODO decorelate. */
      data_.dimensions[SAMPLING_TIME][0] = r[0];
      data_.dimensions[SAMPLING_CLOSURE][0] = r[1];
      data_.dimensions[SAMPLING_RAYTRACE_X][0] = r[0];
    }
    {
      double r[2], offset[2] = {0, 0};
      uint32_t primes[2] = {5, 7};
      BLI_halton_2d(primes, offset, sample_, r);
      data_.dimensions[SAMPLING_LENS_U][0] = r[0];
      data_.dimensions[SAMPLING_LENS_V][0] = r[1];
      /* TODO decorelate. */
      data_.dimensions[SAMPLING_LIGHTPROBE][0] = r[0];
      data_.dimensions[SAMPLING_TRANSPARENCY][0] = r[1];
    }
    {
      /* Using leaped halton sequence so we can reused the same primes as lens. */
      double r[3], offset[3] = {0, 0, 0};
      uint64_t leap = 11;
      uint32_t primes[3] = {5, 4, 7};
      BLI_halton_3d(primes, offset, (sample_ - 1) * leap, r);
      data_.dimensions[SAMPLING_SHADOW_U][0] = r[0];
      data_.dimensions[SAMPLING_SHADOW_V][0] = r[1];
      data_.dimensions[SAMPLING_SHADOW_W][0] = r[2];
      /* TODO decorelate. */
      data_.dimensions[SAMPLING_RAYTRACE_U][0] = r[0];
      data_.dimensions[SAMPLING_RAYTRACE_V][0] = r[1];
      data_.dimensions[SAMPLING_RAYTRACE_W][0] = r[2];
    }
    {
      /* Using leaped halton sequence so we can reused the same primes. */
      double r[2], offset[2] = {0, 0};
      uint64_t leap = 5;
      uint32_t primes[2] = {2, 3};
      BLI_halton_2d(primes, offset, (sample_ - 1) * leap, r);
      data_.dimensions[SAMPLING_SHADOW_X][0] = r[0];
      data_.dimensions[SAMPLING_SHADOW_Y][0] = r[1];
      /* TODO decorelate. */
      data_.dimensions[SAMPLING_SSS_U][0] = r[0];
      data_.dimensions[SAMPLING_SSS_V][0] = r[1];
    }

    data_.push_update();
    sample_++;

    reset_ = false;
  }

  /**
   * Getters
   **/
  /* Returns current, 1 based, scene sample index. */
  uint64_t sample_get(void) const
  {
    return sample_;
  }
  /* Returns blend factor to apply to film to have a smooth transition instead of flickering
   * for the first samples of random shadows. */
  float viewport_smoothing_opacity_factor_get(void) const
  {
    return (sample_ == 1 || viewport_smoothing_duration == 0) ?
               1.0f :
               square_f(clamp_f((sample_viewport_ - viewport_smoothing_start) /
                                    float(viewport_smoothing_duration),
                                0.0f,
                                1.0f));
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
  const GPUUniformBuf *ubo_get(void) const
  {
    return data_.ubo_get();
  }
  /* Returns a pseudo random number in [0..1] range. Each dimension are uncorrelated. */
  float rng_get(eSamplingDimension dimension) const
  {
    return data_.dimensions[dimension][0];
  }
  /* Returns true if rendering has finished. */
  bool finished(void) const
  {
    return (sample_ > sample_count_);
  }
  /* Returns true if viewport smoothing and sampling has finished. */
  bool finished_viewport(void) const
  {
    return finished() &&
           (sample_viewport_ > (viewport_smoothing_start + viewport_smoothing_duration));
  }
  /* Viewport Only: Function to call to notify something in the scene changed.
   * This will reset accumulation. Do not call after end_sync() or during sample rendering. */
  void reset(void)
  {
    reset_ = true;
  }
  /* Viewport Only: true if an update happened in the scene and accumulation needs reset. */
  bool is_reset(void) const
  {
    return reset_;
  }
  /* Return true if we are starting a new motion blur step. We need to run sync agains since
   * depsgraph was updated by MotionBlur::step(). */
  bool do_render_sync(void) const
  {
    return DRW_state_is_image_render() &&
           (((sample_ - 1) % (sample_count_ / motion_blur_steps_)) == 0);
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

  /* Creates a discrete cumulative distribution function table from a given curvemapping.
   * Output cdf vector is expected to already be sized according to the wanted resolution. */
  static void cdf_from_curvemapping(const CurveMapping &curve, Vector<float> &cdf)
  {
    BLI_assert(cdf.size() > 1);
    cdf[0] = 0.0f;
    /* Actual CDF evaluation. */
    for (int u : cdf.index_range()) {
      float x = (float)(u + 1) / (float)(cdf.size() - 1);
      cdf[u + 1] = cdf[u] + BKE_curvemapping_evaluateF(&curve, 0, x);
    }
    /* Normalize the CDF. */
    for (int u : cdf.index_range()) {
      cdf[u] /= cdf.last();
    }
    /* Just to make sure. */
    cdf.last() = 1.0f;
  }

  /* Inverts a cumulative distribution function.
   * Output vector is expected to already be sized according to the wanted resolution. */
  static void cdf_invert(Vector<float> &cdf, Vector<float> &inverted_cdf)
  {
    for (int u : inverted_cdf.index_range()) {
      float x = (float)u / (float)(inverted_cdf.size() - 1);
      for (int i : cdf.index_range()) {
        if (i == cdf.size() - 1) {
          inverted_cdf[u] = 1.0f;
        }
        else if (cdf[i] >= x) {
          float t = (x - cdf[i]) / (cdf[i + 1] - cdf[i]);
          inverted_cdf[u] = ((float)i + t) / (float)(cdf.size() - 1);
          break;
        }
      }
    }
  }

  /**
   * Special ball distribution:
   * Point are distributed in a way that when they are orthogonally
   * projected into any plane, the resulting distribution is (close to)
   * a uniform disc distribution.
   */
  vec3 sample_ball(const float rand[3])
  {
    vec3 sample;
    sample.z = rand[0] * 2.0f - 1.0f; /* cos theta */

    float r = sqrtf(fmaxf(0.0f, 1.0f - square_f(sample.z))); /* sin theta */

    float omega = rand[1] * 2.0f * M_PI;
    sample.x = r * cosf(omega);
    sample.y = r * sinf(omega);

    sample *= sqrtf(sqrtf(rand[2]));
    return sample;
  }

  vec2 sample_disk(const float rand[2])
  {
    float omega = rand[1] * 2.0f * M_PI;
    return sqrtf(rand[0]) * vec2(cosf(omega), sinf(omega));
  }
};

}  // namespace blender::eevee
