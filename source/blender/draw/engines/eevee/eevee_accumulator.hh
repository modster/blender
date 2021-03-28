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
 * An accumulator is a fullscreen buffer (usually at output resolution)
 * that will be able to accumulate sample in any distorted projection
 * using a pixel filter.
 *
 * Input needs to be jittered so that the filter converges to the right result.
 */

#pragma once

#include "GPU_framebuffer.h"
#include "GPU_texture.h"

#include "DRW_render.h"

#include "eevee_camera.hh"
#include "eevee_shaders.hh"

typedef struct EEVEE_AccumulatorParameters {
  int res[2] = {-1, -1};
  float filter_size = 1.0f;
  eEEVEECameraType projection = ORTHO;
} EEVEE_AccumulatorParameters;

inline bool operator==(const EEVEE_AccumulatorParameters &a, const EEVEE_AccumulatorParameters &b)
{
  return equals_v2v2_int(a.res, b.res) && (a.filter_size == b.filter_size) &&
         (a.projection == b.projection);
}

inline bool operator!=(const EEVEE_AccumulatorParameters &a, const EEVEE_AccumulatorParameters &b)
{
  return !(a == b);
}

typedef struct EEVEE_Accumulator {
 public:
  const EEVEE_AccumulatorParameters parameters;

 private:
  /** Owned resources. */
  GPUFrameBuffer *accumulation_fb_ = nullptr;
  GPUTexture *data_tx_ = nullptr;
  GPUTexture *weight_tx_ = nullptr;

  DRWPass *clear_ps_ = nullptr;
  DRWPass *accumulate_ps_ = nullptr;
  DRWPass *resolve_ps_ = nullptr;

  /** Shader parameter, not allocated. */
  GPUTexture *input_tx_;
  /** ViewProjection matrix used to render the input. */
  float *input_persmat_;
  /** ViewProjection matrix Inverse used to render the input. */
  float *input_persinv_;
  /** Number of time the accumulate method was called. */
  uint32_t sample_count_ = 0;

  EEVEE_Shaders &shaders_;

  const char *name_;

 public:
  EEVEE_Accumulator(EEVEE_Shaders &shaders,
                    const char *name,
                    eGPUTextureFormat format,
                    EEVEE_AccumulatorParameters &params)

      : parameters(params), shaders_(shaders), name_(name)
  {
    char full_name[32];

    SNPRINTF(full_name, "Accum.%s.data", name_);
    data_tx_ = GPU_texture_create_2d(name, UNPACK2(parameters.res), 1, format, nullptr);
    /* TODO(fclem) The weight texture could be shared between all similar accumulators. */
    SNPRINTF(full_name, "Accum.%s.weight", name_);
    weight_tx_ = GPU_texture_create_2d(full_name, UNPACK2(parameters.res), 1, GPU_R16F, nullptr);

    GPU_framebuffer_ensure_config(&accumulation_fb_,
                                  {
                                      GPU_ATTACHMENT_NONE,
                                      GPU_ATTACHMENT_TEXTURE(data_tx_),
                                      GPU_ATTACHMENT_TEXTURE(weight_tx_),
                                  });
  }

  ~EEVEE_Accumulator(void)
  {
    GPU_framebuffer_free(accumulation_fb_);
    GPU_texture_free(data_tx_);
    GPU_texture_free(weight_tx_);
  }

  void init(void)
  {
    char full_name[32];
    eGPUSamplerState no_filter = GPU_SAMPLER_DEFAULT;
    {
      SNPRINTF(full_name, "Accum.%s.Accumulate", name_);
      DRWState accum_state = DRW_STATE_WRITE_COLOR | DRW_STATE_BLEND_ADD_FULL;
      accumulate_ps_ = DRW_pass_create(full_name, accum_state);
      GPUShader *sh = shaders_.static_shader_get(ACCUMULATOR_ACCUMULATE);
      DRWShadingGroup *grp = DRW_shgroup_create(sh, accumulate_ps_);
      DRW_shgroup_uniform_texture_ref_ex(grp, "inputTexture", &input_tx_, no_filter);
      DRW_shgroup_uniform_int_copy(grp, "projectionType", parameters.projection);
      DRW_shgroup_uniform_float_copy(grp, "filterSize", parameters.filter_size);
      DRW_shgroup_call_procedural_triangles(grp, NULL, 1);

      /* NOTE: Accumulation needs an additive blend mode, but the first iteration
       * is always going to override the buffer content. */
      SNPRINTF(full_name, "Accum.%s.AccumulateClear", name_);
      DRWState clear_state = DRW_STATE_WRITE_COLOR;
      clear_ps_ = DRW_pass_create_instance(full_name, accumulate_ps_, clear_state);
    }
    {
      SNPRINTF(full_name, "Accum.%s.Resolve", name_);
      resolve_ps_ = DRW_pass_create(full_name, DRW_STATE_WRITE_COLOR);
      GPUShader *sh = shaders_.static_shader_get(ACCUMULATOR_RESOLVE);
      DRWShadingGroup *grp = DRW_shgroup_create(sh, resolve_ps_);
      DRW_shgroup_uniform_texture_ref_ex(grp, "dataTexture", &data_tx_, no_filter);
      DRW_shgroup_uniform_texture_ref_ex(grp, "weightTexture", &weight_tx_, no_filter);
      DRW_shgroup_call_procedural_triangles(grp, NULL, 1);
    }
    /* TEST */
    sample_count_ = 0;
  }

  void accumulate(GPUTexture *input, const DRWView *view)
  {
    float persmat[4][4], persinv[4][4];
    DRW_view_persmat_get(view, persmat, false);
    DRW_view_persmat_get(view, persinv, true);
    input_persinv_ = &persinv[0][0];
    input_persmat_ = &persmat[0][0];
    input_tx_ = input;

    GPU_framebuffer_bind(accumulation_fb_);

    if (sample_count_ == 0) {
      DRW_draw_pass(clear_ps_);
    }
    else {
      DRW_draw_pass(accumulate_ps_);
    }
    sample_count_++;
  }

  void resolve_onto(GPUFrameBuffer *target)
  {
    BLI_assert(sample_count_ > 0);

    GPU_framebuffer_bind(target);

    DRW_draw_pass(resolve_ps_);
  }

  void read_to_memory(float *data)
  {
    /* TODO(fclem) implement. */
    (void)data;
  }

} EEVEE_Accumulator;
