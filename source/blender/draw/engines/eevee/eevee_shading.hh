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
/** \name Passes
 * \{ */

class ForwardPass {
 private:
  Instance &inst_;

  DRWPass *opaque_ps_ = nullptr;
  DRWPass *light_additional_ps_ = nullptr;

 public:
  ForwardPass(Instance &inst) : inst_(inst){};

  void sync(void);
  void surface_add(Object *ob, Material *mat, int matslot);
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
  Texture diffuse_tx = Texture("GbufferDiffuse");
  Texture reflection_tx = Texture("GbufferReflection");
  Texture refraction_tx = Texture("GbufferRefraction");
  Texture volume_tx = Texture("GbufferVolume");
  Texture emission_tx = Texture("GbufferEmission");
  Texture transparency_tx = Texture("GbufferTransparency");

  Framebuffer framebuffer = Framebuffer("Gbuffer");
  /* Pointer to the view's depth buffer. */
  GPUTexture *depth_tx = nullptr;

  void sync(GPUTexture *depth_tx_)
  {
    depth_tx = depth_tx_;
    diffuse_tx.sync_tmp();
    reflection_tx.sync_tmp();
    refraction_tx.sync_tmp();
    volume_tx.sync_tmp();
    emission_tx.sync_tmp();
    transparency_tx.sync_tmp();
  }

  void bind(eClosureBits closures_used)
  {
    UNUSED_VARS(closures_used);
    ivec2 extent = {GPU_texture_width(depth_tx), GPU_texture_height(depth_tx)};

    /* TODO Allocate only the one we need. */
    /* TODO Reuse for different config. */
    /* TODO Allocate only GPU_RG32UI for diffuse if no sss is needed. */
    diffuse_tx.acquire_tmp(UNPACK2(extent), GPU_RGBA32UI, this);
    reflection_tx.acquire_tmp(UNPACK2(extent), GPU_RG32UI, this);
    // refraction_tx.acquire_tmp(UNPACK2(extent), GPU_RGBA32UI, this);
    // volume_tx.acquire_tmp(UNPACK2(extent), GPU_RGBA32UI, this);
    // emission_tx.acquire_tmp(UNPACK2(extent), GPU_RGBA16F, this);
    // transparency_tx.acquire_tmp(UNPACK2(extent), GPU_RGBA16, this);

    framebuffer.ensure(GPU_ATTACHMENT_TEXTURE(depth_tx),
                       GPU_ATTACHMENT_TEXTURE(diffuse_tx),
                       GPU_ATTACHMENT_TEXTURE(reflection_tx));
    GPU_framebuffer_bind(framebuffer);
  }

  void render_end(void)
  {
    diffuse_tx.release_tmp();
    reflection_tx.release_tmp();
    refraction_tx.release_tmp();
    volume_tx.release_tmp();
    emission_tx.release_tmp();
    transparency_tx.release_tmp();
  }
};

class DeferredLayer {
 private:
  Instance &inst_;

  GPUTexture *input_diffuse_data_tx_ = nullptr;
  GPUTexture *input_depth_tx_ = nullptr;

  DRWPass *gbuffer_ps_ = nullptr;

 public:
  DeferredLayer(Instance &inst) : inst_(inst){};

  void sync(void);
  void surface_add(Object *ob);
  void render(GBuffer &gbuffer, GPUFrameBuffer *view_fb);
};

class DeferredPass {
  friend DeferredLayer;

 private:
  Instance &inst_;

  /* Gbuffer filling passes. We could have an arbitrary number of them but for now we just have
   * a harcoded number of them. */
  DeferredLayer opaque_ps_;
  DeferredLayer refraction_ps_;
  DeferredLayer volumetric_ps_;

  DRWPass *eval_diffuse_ps_ = nullptr;

  /* References only. */
  GPUTexture *input_diffuse_data_tx_ = nullptr;
  GPUTexture *input_reflection_data_tx_ = nullptr;
  GPUTexture *input_depth_tx_ = nullptr;

 public:
  DeferredPass(Instance &inst)
      : inst_(inst), opaque_ps_(inst), refraction_ps_(inst), volumetric_ps_(inst){};

  void sync(void);
  void surface_add(Object *ob);
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

  DeferredPass deferred;
  ForwardPass forward;
  ShadowPass shadow;
  VelocityPass velocity;

  CullingDebugPass debug_culling;

  UtilityTexture utility_tx;

 public:
  ShadingPasses(Instance &inst)
      : light_culling(inst),
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
};

/** \} */

}  // namespace blender::eevee