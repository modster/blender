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
 * Shading passes contain drawcalls specific to shading pipelines.
 * They are to be shared across views.
 * This file is only for shading passes. Other passes are declared in their own module.
 */

#pragma once

#include "DRW_render.h"

#include "eevee_lut.h"

#include "eevee_culling.hh"
#include "eevee_shadow.hh"
#include "eevee_velocity.hh"

namespace blender::eevee {

class Instance;

/* -------------------------------------------------------------------- */
/** \name Background Pass
 *
 * Render world values.
 * \{ */

class BackgroundPass {
 private:
  Instance &inst_;

  DRWPass *background_ps_ = nullptr;

 public:
  BackgroundPass(Instance &inst) : inst_(inst){};

  void sync(GPUMaterial *gpumat, GPUTexture *loodev_tx = nullptr);
  void render(void);
};

/** \} */

/* -------------------------------------------------------------------- */
/** \name Forward Pass
 *
 * Handles alpha blended surfaces and NPR materials (using Closure to RGBA).
 * \{ */

class ForwardPass {
 private:
  Instance &inst_;

  DRWPass *prepass_ps_ = nullptr;
  DRWPass *prepass_culled_ps_ = nullptr;
  DRWPass *opaque_ps_ = nullptr;
  DRWPass *opaque_culled_ps_ = nullptr;
  DRWPass *transparent_ps_ = nullptr;

 public:
  ForwardPass(Instance &inst) : inst_(inst){};

  void sync(void);

  DRWShadingGroup *material_add(::Material *blender_mat, GPUMaterial *gpumat)
  {
    return (GPU_material_flag_get(gpumat, GPU_MATFLAG_TRANSPARENT)) ?
               material_transparent_add(blender_mat, gpumat) :
               material_opaque_add(blender_mat, gpumat);
  }

  DRWShadingGroup *prepass_add(::Material *blender_mat, GPUMaterial *gpumat)
  {
    return (GPU_material_flag_get(gpumat, GPU_MATFLAG_TRANSPARENT)) ?
               prepass_transparent_add(blender_mat, gpumat) :
               prepass_opaque_add(blender_mat, gpumat);
  }

  DRWShadingGroup *material_opaque_add(::Material *blender_mat, GPUMaterial *gpumat);
  DRWShadingGroup *prepass_opaque_add(::Material *blender_mat, GPUMaterial *gpumat);
  DRWShadingGroup *material_transparent_add(::Material *blender_mat, GPUMaterial *gpumat);
  DRWShadingGroup *prepass_transparent_add(::Material *blender_mat, GPUMaterial *gpumat);
  void render(void);
};

/** \} */

/* -------------------------------------------------------------------- */
/** \name Deferred lighting.
 * \{ */

enum eClosureBits {
  CLOSURE_DIFFUSE = 1 << 0,
  CLOSURE_SSS = 1 << 1,
  CLOSURE_REFLECTION = 1 << 2,
  CLOSURE_REFRACTION = 1 << 3,
  CLOSURE_VOLUME = 1 << 4,
  CLOSURE_EMISSION = 1 << 5,
  CLOSURE_TRANSPARENCY = 1 << 6,
};

struct GBuffer {
  Texture transmit_color_tx = Texture("GbufferTransmitColor");
  Texture transmit_normal_tx = Texture("GbufferTransmitNormal");
  Texture transmit_data_tx = Texture("GbufferTransmitData");
  Texture reflect_color_tx = Texture("GbufferReflectionColor");
  Texture reflect_normal_tx = Texture("GbufferReflectionNormal");
  Texture volume_tx = Texture("GbufferVolume");
  Texture emission_tx = Texture("GbufferEmission");
  Texture transparency_tx = Texture("GbufferTransparency");

  Framebuffer gbuffer_fb = Framebuffer("Gbuffer");
  Framebuffer volume_fb = Framebuffer("VolumeHeterogeneous");

  Texture holdout_tx = Texture("HoldoutRadiance");

  Framebuffer holdout_fb = Framebuffer("Holdout");

  Texture depth_behind_tx = Texture("DepthBehind");
  Texture depth_copy_tx = Texture("DepthCopy");

  Framebuffer depth_behind_fb = Framebuffer("DepthCopy");
  Framebuffer depth_copy_fb = Framebuffer("DepthCopy");

  /* Owner of this GBuffer. Used to query temp textures. */
  void *owner;

  /* Pointer to the view's buffers. */
  GPUTexture *depth_tx = nullptr;
  GPUTexture *combined_tx = nullptr;
  int layer = -1;

  void sync(GPUTexture *depth_tx_, GPUTexture *combined_tx_, void *owner_, int layer_ = -1)
  {
    owner = owner_;
    depth_tx = depth_tx_;
    combined_tx = combined_tx_;
    layer = layer_;
    transmit_color_tx.sync_tmp();
    transmit_normal_tx.sync_tmp();
    transmit_data_tx.sync_tmp();
    reflect_color_tx.sync_tmp();
    reflect_normal_tx.sync_tmp();
    volume_tx.sync_tmp();
    emission_tx.sync_tmp();
    transparency_tx.sync_tmp();
    holdout_tx.sync_tmp();
    depth_behind_tx.sync_tmp();
    depth_copy_tx.sync_tmp();
  }

  void bind(eClosureBits closures_used)
  {
    ivec2 extent = {GPU_texture_width(depth_tx), GPU_texture_height(depth_tx)};

    /* TODO Reuse for different config. */
    if (closures_used & (CLOSURE_DIFFUSE | CLOSURE_SSS | CLOSURE_REFRACTION)) {
      transmit_color_tx.acquire_tmp(UNPACK2(extent), GPU_R11F_G11F_B10F, owner);
    }
    if (closures_used & (CLOSURE_SSS | CLOSURE_REFRACTION)) {
      transmit_normal_tx.acquire_tmp(UNPACK2(extent), GPU_RGBA16F, owner);
      transmit_data_tx.acquire_tmp(UNPACK2(extent), GPU_R11F_G11F_B10F, owner);
    }
    else if (closures_used & CLOSURE_DIFFUSE) {
      transmit_normal_tx.acquire_tmp(UNPACK2(extent), GPU_RG16F, owner);
    }

    if (closures_used & CLOSURE_DIFFUSE) {
      reflect_color_tx.acquire_tmp(UNPACK2(extent), GPU_R11F_G11F_B10F, owner);
      reflect_normal_tx.acquire_tmp(UNPACK2(extent), GPU_RGBA16, owner);
    }

    if (closures_used & CLOSURE_VOLUME) {
      /* TODO(fclem): This is killing performance.
       * Idea: use interleaved data pattern to fill only a 32bpp buffer. */
      volume_tx.acquire_tmp(UNPACK2(extent), GPU_RGBA32UI, owner);
    }

    if (closures_used & CLOSURE_EMISSION) {
      emission_tx.acquire_tmp(UNPACK2(extent), GPU_R11F_G11F_B10F, owner);
    }

    if (closures_used & CLOSURE_TRANSPARENCY) {
      /* TODO(fclem): Speedup by using Dithered holdout and GPU_RGB10_A2. */
      transparency_tx.acquire_tmp(UNPACK2(extent), GPU_RGBA16, owner);
    }

    holdout_tx.acquire_tmp(UNPACK2(extent), GPU_R11F_G11F_B10F, owner);
    depth_behind_tx.acquire_tmp(UNPACK2(extent), GPU_DEPTH24_STENCIL8, owner);
    depth_copy_tx.acquire_tmp(UNPACK2(extent), GPU_DEPTH24_STENCIL8, owner);

    /* Layer attachement also works with cubemap. */
    gbuffer_fb.ensure(GPU_ATTACHMENT_TEXTURE_LAYER(depth_tx, layer),
                      GPU_ATTACHMENT_TEXTURE(transmit_color_tx),
                      GPU_ATTACHMENT_TEXTURE(transmit_normal_tx),
                      GPU_ATTACHMENT_TEXTURE(transmit_data_tx),
                      GPU_ATTACHMENT_TEXTURE(reflect_color_tx),
                      GPU_ATTACHMENT_TEXTURE(reflect_normal_tx),
                      GPU_ATTACHMENT_TEXTURE(volume_tx),
                      GPU_ATTACHMENT_TEXTURE(emission_tx),
                      GPU_ATTACHMENT_TEXTURE(transparency_tx));
    GPU_framebuffer_bind(gbuffer_fb);
    GPU_framebuffer_clear_stencil(gbuffer_fb, 0x0);
  }

  void bind_volume(void)
  {
    volume_fb.ensure(GPU_ATTACHMENT_TEXTURE(depth_tx),
                     GPU_ATTACHMENT_TEXTURE(volume_tx),
                     GPU_ATTACHMENT_TEXTURE(transparency_tx));
    GPU_framebuffer_bind(volume_fb);
  }

  void bind_holdout(void)
  {
    holdout_fb.ensure(GPU_ATTACHMENT_NONE, GPU_ATTACHMENT_TEXTURE(holdout_tx));
    GPU_framebuffer_bind(holdout_fb);
  }

  void copy_depth_behind(void)
  {
    depth_behind_fb.ensure(GPU_ATTACHMENT_TEXTURE(depth_behind_tx));
    GPU_framebuffer_bind(depth_behind_fb);

    GPU_framebuffer_blit(gbuffer_fb, 0, depth_behind_fb, 0, GPU_DEPTH_BIT);
  }

  void copy_depth(void)
  {
    depth_copy_fb.ensure(GPU_ATTACHMENT_TEXTURE(depth_copy_tx));
    GPU_framebuffer_bind(depth_copy_fb);

    GPU_framebuffer_blit(gbuffer_fb, 0, depth_copy_fb, 0, GPU_DEPTH_BIT);
  }

  void render_end(void)
  {
    transmit_color_tx.release_tmp();
    transmit_normal_tx.release_tmp();
    transmit_data_tx.release_tmp();
    reflect_color_tx.release_tmp();
    reflect_normal_tx.release_tmp();
    volume_tx.release_tmp();
    emission_tx.release_tmp();
    transparency_tx.release_tmp();
    holdout_tx.release_tmp();
    depth_behind_tx.release_tmp();
    depth_copy_tx.release_tmp();
  }
};

class DeferredLayer {
 private:
  Instance &inst_;

  /* TODO */
  // GPUTexture *input_emission_data_tx_ = nullptr;
  // GPUTexture *input_diffuse_data_tx_ = nullptr;
  // GPUTexture *input_depth_tx_ = nullptr;

  DRWPass *prepass_ps_ = nullptr;
  DRWPass *prepass_culled_ps_ = nullptr;
  DRWPass *gbuffer_ps_ = nullptr;
  DRWPass *gbuffer_culled_ps_ = nullptr;
  DRWPass *volume_ps_ = nullptr;

 public:
  DeferredLayer(Instance &inst) : inst_(inst){};

  void sync(void);
  DRWShadingGroup *material_add(::Material *blender_mat, GPUMaterial *gpumat);
  DRWShadingGroup *prepass_add(::Material *blender_mat, GPUMaterial *gpumat);
  void volume_add(Object *ob);
  void render(GBuffer &gbuffer, GPUFrameBuffer *view_fb);
};

class DeferredPass {
  friend DeferredLayer;

 private:
  Instance &inst_;

  /* Gbuffer filling passes. We could have an arbitrary number of them but for now we just have
   * a harcoded number of them. */
  DeferredLayer opaque_layer_;
  DeferredLayer refraction_layer_;
  DeferredLayer volumetric_layer_;

  DRWPass *eval_diffuse_ps_ = nullptr;
  DRWPass *eval_transparency_ps_ = nullptr;
  DRWPass *eval_holdout_ps_ = nullptr;
  // DRWPass *eval_volume_heterogeneous_ps_ = nullptr;
  DRWPass *eval_volume_homogeneous_ps_ = nullptr;

  /* References only. */
  GPUTexture *input_combined_tx = nullptr;
  GPUTexture *input_depth_behind_tx_ = nullptr;
  GPUTexture *input_depth_tx_ = nullptr;
  GPUTexture *input_emission_data_tx_ = nullptr;
  GPUTexture *input_transmit_color_tx_ = nullptr;
  GPUTexture *input_transmit_normal_tx_ = nullptr;
  GPUTexture *input_transmit_data_tx_ = nullptr;
  GPUTexture *input_reflect_color_tx_ = nullptr;
  GPUTexture *input_reflect_normal_tx_ = nullptr;
  GPUTexture *input_transparency_data_tx_ = nullptr;
  GPUTexture *input_volume_data_tx_ = nullptr;
  // GPUTexture *input_volume_radiance_tx_ = nullptr;
  // GPUTexture *input_volume_transmittance_tx_ = nullptr;

 public:
  DeferredPass(Instance &inst)
      : inst_(inst), opaque_layer_(inst), refraction_layer_(inst), volumetric_layer_(inst){};

  void sync(void);
  DRWShadingGroup *material_add(::Material *material, GPUMaterial *gpumat);
  DRWShadingGroup *prepass_add(::Material *material, GPUMaterial *gpumat);
  void volume_add(Object *ob);
  void render(GBuffer &gbuffer, GPUFrameBuffer *view_fb);
};

/** \} */

/* -------------------------------------------------------------------- */
/** \name Utility texture
 *
 * 64x64 2D array texture containing LUT tables and blue noises.
 * \{ */

class UtilityTexture : public Texture {
  struct Layer {
    float data[UTIL_TEX_SIZE * UTIL_TEX_SIZE][4];
  };

  static constexpr int lut_size = UTIL_TEX_SIZE;
  static constexpr int lut_size_sqr = lut_size * lut_size;
  static constexpr int layer_count = 4 + UTIL_BTDF_LAYER_COUNT;

 public:
  UtilityTexture()
      : Texture("UtilityTx", lut_size, lut_size, layer_count, 1, GPU_RGBA16F, nullptr, true)
  {
#ifdef RUNTIME_LUT_CREATION
    float *bsdf_ggx_lut = EEVEE_lut_update_ggx_brdf(lut_size);
    float(*btdf_ggx_lut)[lut_size_sqr * 2] = (float(*)[lut_size_sqr * 2])
        EEVEE_lut_update_ggx_btdf(lut_size, UTIL_BTDF_LAYER_COUNT);
#else
    const float *bsdf_ggx_lut = bsdf_split_sum_ggx;
    const float(*btdf_ggx_lut)[lut_size_sqr * 2] = btdf_split_sum_ggx;
#endif

    Vector<Layer> data(layer_count);
    {
      Layer &layer = data[UTIL_BLUE_NOISE_LAYER];
      memcpy(layer.data, blue_noise, sizeof(layer));
    }
    {
      Layer &layer = data[UTIL_LTC_MAT_LAYER];
      memcpy(layer.data, ltc_mat_ggx, sizeof(layer));
    }
    {
      Layer &layer = data[UTIL_LTC_MAG_LAYER];
      for (auto i : IndexRange(lut_size_sqr)) {
        layer.data[i][0] = bsdf_ggx_lut[i * 2 + 0];
        layer.data[i][1] = bsdf_ggx_lut[i * 2 + 1];
        layer.data[i][2] = ltc_mag_ggx[i * 2 + 0];
        layer.data[i][3] = ltc_mag_ggx[i * 2 + 1];
      }
      BLI_assert(UTIL_LTC_MAG_LAYER == UTIL_BSDF_LAYER);
    }
    {
      Layer &layer = data[UTIL_DISK_INTEGRAL_LAYER];
      for (auto i : IndexRange(lut_size_sqr)) {
        layer.data[i][UTIL_DISK_INTEGRAL_COMP] = ltc_disk_integral[i];
      }
    }
    {
      for (auto layer_id : IndexRange(16)) {
        Layer &layer = data[3 + layer_id];
        for (auto i : IndexRange(lut_size_sqr)) {
          layer.data[i][0] = btdf_ggx_lut[layer_id][i * 2 + 0];
          layer.data[i][1] = btdf_ggx_lut[layer_id][i * 2 + 1];
        }
      }
    }
    GPU_texture_update_mipmap(*this, 0, GPU_DATA_FLOAT, data.data());
  }

  ~UtilityTexture(){};
};

/** \} */

/* -------------------------------------------------------------------- */
/** \name ShadingPasses
 *
 * \{ */

/**
 * Shading passes. Shared between views. Objects will subscribe to one of them.
 */
class ShadingPasses {
 public:
  CullingLightPass light_culling;

  BackgroundPass background;
  DeferredPass deferred;
  ForwardPass forward;
  ShadowPass shadow;
  VelocityPass velocity;

  CullingDebugPass debug_culling;

  UtilityTexture utility_tx;

 public:
  ShadingPasses(Instance &inst)
      : light_culling(inst),
        background(inst),
        deferred(inst),
        forward(inst),
        shadow(inst),
        velocity(inst),
        debug_culling(inst){};

  void sync()
  {
    light_culling.sync();

    deferred.sync();
    forward.sync();
    shadow.sync();
    velocity.sync();

    debug_culling.sync();
  }

  DRWShadingGroup *material_add(::Material *blender_mat,
                                GPUMaterial *gpumat,
                                eMaterialPipeline pipeline_type)
  {
    switch (pipeline_type) {
      case MAT_PIPE_DEFERRED_PREPASS:
        return deferred.prepass_add(blender_mat, gpumat);
      case MAT_PIPE_FORWARD_PREPASS:
        return forward.prepass_add(blender_mat, gpumat);
      case MAT_PIPE_DEFERRED:
        return deferred.material_add(blender_mat, gpumat);
      case MAT_PIPE_FORWARD:
        return forward.material_add(blender_mat, gpumat);
      case MAT_PIPE_VOLUME:
        /* TODO(fclem) volume pass. */
        return nullptr;
      case MAT_PIPE_SHADOW:
        return shadow.material_add(blender_mat, gpumat);
    }
    return nullptr;
  }
};

/** \} */

}  // namespace blender::eevee