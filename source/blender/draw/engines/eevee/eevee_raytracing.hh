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

#include "eevee_gbuffer.hh"
#include "eevee_shader_shared.hh"

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
  RaytraceDataBuf diffuse_data_;

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
  const GPUUniformBuf *diffuse_ubo_get(void) const
  {
    return diffuse_data_.ubo_get();
  }

  bool enabled(void) const
  {
    return enabled_;
  }
};

/** \} */

/* -------------------------------------------------------------------- */
/** \name Raytracing Buffers
 *
 * Contain persistent data used for temporal denoising. Similar to \class GBuffer but only contains
 * persistent data.
 * \{ */

struct RaytraceBuffer {
 public:
  DRWPass *denoise_diffuse_ps_ = nullptr;
  DRWPass *denoise_reflection_ps_ = nullptr;
  DRWPass *denoise_refraction_ps_ = nullptr;
  DRWPass *resolve_diffuse_ps_ = nullptr;
  DRWPass *resolve_reflection_ps_ = nullptr;
  DRWPass *resolve_refraction_ps_ = nullptr;
  DRWPass *trace_diffuse_ps_ = nullptr;
  DRWPass *trace_reflection_ps_ = nullptr;
  DRWPass *trace_refraction_ps_ = nullptr;

 private:
  Instance &inst_;

  /* Only allocated if used. */
  Texture diffuse_radiance_tx_ = Texture("DiffuseHistory_A");
  Texture diffuse_radiance_history_tx_ = Texture("DiffuseHistory_B");
  Texture diffuse_variance_tx_ = Texture("DiffuseVariance_A");
  Texture diffuse_variance_history_tx_ = Texture("DiffuseVariance_B");
  Texture reflection_radiance_tx_ = Texture("ReflectionHistory_A");
  Texture reflection_radiance_history_tx_ = Texture("ReflectionHistory_B");
  Texture reflection_variance_tx_ = Texture("ReflectionVariance_A");
  Texture reflection_variance_history_tx_ = Texture("ReflectionVariance_B");
  Texture refraction_radiance_tx_ = Texture("RefractionHistory_A");
  Texture refraction_radiance_history_tx_ = Texture("RefractionHistory_B");
  Texture refraction_variance_tx_ = Texture("RefractionVariance_A");
  Texture refraction_variance_history_tx_ = Texture("RefractionVariance_B");

  /* Reference only. */
  GPUTexture *input_radiance_tx_;
  GPUTexture *input_combined_tx_;
  GPUTexture *input_ray_data_tx_;
  GPUTexture *input_ray_color_tx_;
  GPUTexture *input_hiz_tx_;
  GPUTexture *input_hiz_front_tx_;
  GPUTexture *input_cl_color_tx_;
  GPUTexture *input_cl_normal_tx_;
  GPUTexture *input_cl_data_tx_;
  GPUTexture *input_history_tx_;
  GPUTexture *input_variance_tx_;
  GPUTexture *output_history_tx_;
  GPUTexture *output_variance_tx_;

  RaytraceBufferDataBuf data_;

  ivec2 extent_ = ivec2(0);
  ivec3 dispatch_size_ = ivec3(1);

 public:
  RaytraceBuffer(Instance &inst) : inst_(inst){};
  ~RaytraceBuffer(){};

  void sync(ivec2 extent);

  void trace(eClosureBits closure_type, GBuffer &gbuffer, HiZBuffer &hiz, HiZBuffer &hiz_front);
  void denoise(eClosureBits closure_type);
  void resolve(eClosureBits closure_type, GBuffer &gbuffer);

  GPUTexture *diffuse_radiance_history_get(void)
  {
    ensure_buffer(diffuse_radiance_history_tx_, data_.valid_history_diffuse, GPU_RGBA16F);
    return diffuse_radiance_history_tx_;
  }
  GPUTexture *reflection_radiance_history_get(void)
  {
    ensure_buffer(reflection_radiance_history_tx_, data_.valid_history_reflection, GPU_RGBA16F);
    return reflection_radiance_history_tx_;
  }
  GPUTexture *refraction_radiance_history_get(void)
  {
    ensure_buffer(refraction_radiance_history_tx_, data_.valid_history_refraction, GPU_RGBA16F);
    return refraction_radiance_history_tx_;
  }

  GPUTexture *diffuse_variance_history_get(void)
  {
    ensure_buffer(diffuse_variance_history_tx_, data_.valid_history_diffuse, GPU_R8);
    return diffuse_variance_history_tx_;
  }
  GPUTexture *reflection_variance_history_get(void)
  {
    ensure_buffer(reflection_variance_history_tx_, data_.valid_history_reflection, GPU_R8);
    return reflection_variance_history_tx_;
  }
  GPUTexture *refraction_variance_history_get(void)
  {
    ensure_buffer(refraction_variance_history_tx_, data_.valid_history_refraction, GPU_R8);
    return refraction_variance_history_tx_;
  }

  GPUTexture *diffuse_radiance_get(void)
  {
    ensure_buffer(diffuse_radiance_tx_, data_.valid_history_diffuse, GPU_RGBA16F);
    return diffuse_radiance_tx_;
  }
  GPUTexture *reflection_radiance_get(void)
  {
    ensure_buffer(reflection_radiance_tx_, data_.valid_history_reflection, GPU_RGBA16F);
    return reflection_radiance_tx_;
  }
  GPUTexture *refraction_radiance_get(void)
  {
    ensure_buffer(refraction_radiance_tx_, data_.valid_history_refraction, GPU_RGBA16F);
    return refraction_radiance_tx_;
  }

  GPUTexture *diffuse_variance_get(void)
  {
    ensure_buffer(diffuse_variance_tx_, data_.valid_history_diffuse, GPU_R8);
    return diffuse_variance_tx_;
  }
  GPUTexture *reflection_variance_get(void)
  {
    ensure_buffer(reflection_variance_tx_, data_.valid_history_reflection, GPU_R8);
    return reflection_variance_tx_;
  }
  GPUTexture *refraction_variance_get(void)
  {
    ensure_buffer(refraction_variance_tx_, data_.valid_history_refraction, GPU_R8);
    return refraction_variance_tx_;
  }

  void render_end(const DRWView *view)
  {
    DRW_view_persmat_get(view, data_.history_persmat, false);
    SWAP(Texture, diffuse_radiance_tx_, diffuse_radiance_history_tx_);
    SWAP(Texture, diffuse_variance_tx_, diffuse_variance_history_tx_);
    SWAP(Texture, reflection_radiance_tx_, reflection_radiance_history_tx_);
    SWAP(Texture, reflection_variance_tx_, reflection_variance_history_tx_);
    SWAP(Texture, refraction_radiance_tx_, refraction_radiance_history_tx_);
    SWAP(Texture, refraction_variance_tx_, refraction_variance_history_tx_);
  }

 private:
  void ensure_buffer(Texture &texture, int &valid_history, eGPUTextureFormat format)
  {
    bool was_allocated = texture.ensure(UNPACK2(extent_), 1, format);
    if (was_allocated && valid_history) {
      valid_history = false;
      data_.push_update();
    }
    else if (!was_allocated && !valid_history) {
      valid_history = true;
      data_.push_update();
    }
  }
};

/** \} */

}  // namespace blender::eevee
