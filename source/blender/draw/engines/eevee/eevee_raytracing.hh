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

  /* Persistent buffers. */
  Texture ray_history_diffuse_tx_[2] = {"RayDiffHistory1", "RayDiffHistory2"};
  Texture ray_history_reflect_tx_[2] = {"RayReflHistory1", "RayReflHistory2"};
  Texture ray_history_refract_tx_[2] = {"RayRefrHistory1", "RayRefrHistory2"};
  Texture ray_variance_diffuse_tx_[2] = {"RayDiffVariance1", "RayDiffVariance2"};
  Texture ray_variance_reflect_tx_[2] = {"RayReflVariance1", "RayReflVariance2"};
  Texture ray_variance_refract_tx_[2] = {"RayRefrVariance1", "RayRefrVariance2"};

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
    Texture::swap(ray_history_diffuse_tx_[0], ray_history_diffuse_tx_[1]);
    Texture::swap(ray_history_reflect_tx_[0], ray_history_reflect_tx_[1]);
    Texture::swap(ray_history_refract_tx_[0], ray_history_refract_tx_[1]);
    Texture::swap(ray_variance_diffuse_tx_[0], ray_variance_diffuse_tx_[1]);
    Texture::swap(ray_variance_reflect_tx_[0], ray_variance_reflect_tx_[1]);
    Texture::swap(ray_variance_refract_tx_[0], ray_variance_refract_tx_[1]);
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

  DRWPass *sync_raytrace_passes(const char *name,
                                eShaderType screen_trace_sh,
                                eShaderType denoise_sh,
                                HiZBuffer &hiz_tracing,
                                TextureFromPool &ray_data_tx,
                                TextureFromPool &ray_radiance_tx,
                                Texture &ray_history_src_tx,
                                Texture &ray_history_dst_tx,
                                Texture &ray_variance_src_tx,
                                Texture &ray_variance_dst_tx,
                                TextureFromPool &gbuf_data_tx,
                                TextureFromPool &gbuf_normal_tx,
                                RaytraceDataBuf &ray_data_buf,
                                RaytraceIndirectBuf &dispatch_buf,
                                RaytraceTileBuf &tile_buf);
};

/** \} */

}  // namespace blender::eevee
