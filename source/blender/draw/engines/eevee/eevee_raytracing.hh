/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2021 Blender Foundation.
 */

#pragma once

#include "DRW_render.h"

#include "eevee_gbuffer.hh"
#include "eevee_shader_shared.hh"

namespace blender::eevee {

class Instance;
class DeferredPass;

/* -------------------------------------------------------------------- */
/** \name Raytracing
 * \{ */

class RaytracingModule {
 public:
  RaytraceDataBuf reflection_data;
  RaytraceDataBuf refraction_data;
  RaytraceDataBuf diffuse_data;

 private:
  Instance &inst_;

  bool enabled_ = false;

 public:
  RaytracingModule(Instance &inst) : inst_(inst){};

  void sync(void);

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
  DRWPass *raygen_ps_ = nullptr;
  DRWPass *trace_diffuse_ps_ = nullptr;
  DRWPass *trace_reflect_ps_ = nullptr;
  DRWPass *trace_refract_ps_ = nullptr;

 private:
  Instance &inst_;

  RaytraceBufferDataBuf data_;

  RaytraceIndirectBuf dispatch_diffuse_buf_, dispatch_reflect_buf_, dispatch_refract_buf_;
  RaytraceTileBuf tiles_diffuse_buf_, tiles_reflect_buf_, tiles_refract_buf_;

  TextureFromPool ray_data_diffuse_tx_ = {"RayDiffData"};
  TextureFromPool ray_data_reflect_tx_ = {"RayReflData"};
  TextureFromPool ray_data_refract_tx_ = {"RayRefrData"};
  TextureFromPool ray_radiance_diffuse_tx_ = {"RayDiffRadiance"};
  TextureFromPool ray_radiance_reflect_tx_ = {"RayReflRadiance"};
  TextureFromPool ray_radiance_refract_tx_ = {"RayRefrRadiance"};

  int2 extent_ = int2(0);
  int3 raygen_dispatch_size_ = int3(1);

  /* Reference only. */
  GPUTexture *depth_view_tx_ = nullptr;
  GPUTexture *stencil_view_tx_ = nullptr;

 public:
  RaytraceBuffer(Instance &inst) : inst_(inst){};
  ~RaytraceBuffer(){};

  void sync(int2 extent);

  void trace(eClosureBits closure_type, Texture &depth_buffer, DeferredPass &deferred_pass);
  void release_tmp();

  void render_end(const DRWView *view)
  {
    using draw::Texture;
    DRW_view_persmat_get(view, data_.history_persmat.ptr(), false);
    // Texture::swap(diffuse_radiance_tx_, diffuse_radiance_history_tx_);
    // Texture::swap(diffuse_variance_tx_, diffuse_variance_history_tx_);
    // Texture::swap(reflection_radiance_tx_, reflection_radiance_history_tx_);
    // Texture::swap(reflection_variance_tx_, reflection_variance_history_tx_);
    // Texture::swap(refraction_radiance_tx_, refraction_radiance_history_tx_);
    // Texture::swap(refraction_variance_tx_, refraction_variance_history_tx_);
  }

 private:
  void ensure_buffer(Texture &texture, int &valid_history, eGPUTextureFormat format)
  {
    bool was_allocated = texture.ensure_2d(format, extent_);
    if (was_allocated && valid_history) {
      valid_history = false;
      data_.push_update();
    }
    else if (!was_allocated && !valid_history) {
      valid_history = true;
      data_.push_update();
    }
  }

  DRWPass *sync_raytrace_pass(const char *name,
                              eShaderType screen_trace_sh,
                              HiZBuffer &hiz_tracing,
                              TextureFromPool &ray_data_tx,
                              TextureFromPool &ray_radiance_tx,
                              RaytraceIndirectBuf &dispatch_buf,
                              RaytraceTileBuf &tile_buf);
};

/** \} */

}  // namespace blender::eevee
