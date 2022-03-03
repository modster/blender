/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2021 Blender Foundation.
 */

/** \file
 * \ingroup eevee
 *
 * Gbuffer layout used for deferred shading pipeline.
 */

#pragma once

#include "DRW_render.h"

#include "eevee_wrapper.hh"

namespace blender::eevee {

class Instance;

/* -------------------------------------------------------------------- */
/** \name Gbuffer
 *
 * Fullscreen textures containing geometric, surface and volume data.
 * Used by deferred shading layers. Only one gbuffer is allocated per view
 * and is reused for each deferred layer. This is why there can only be temporary
 * texture inside it.
 * \{ */

/** NOTE: Theses are used as stencil bits. So we are limited to 8bits. */
enum eClosureBits {
  CLOSURE_DIFFUSE = 1 << 0,
  CLOSURE_SSS = 1 << 1,
  CLOSURE_REFLECTION = 1 << 2,
  CLOSURE_REFRACTION = 1 << 3,
  CLOSURE_AMBIENT_OCCLUSION = 1 << 5,
  /* Non-stencil bits. */
  CLOSURE_TRANSPARENCY = 1 << 8,
  CLOSURE_EMISSION = 1 << 9,
  CLOSURE_HOLDOUT = 1 << 10,
  CLOSURE_VOLUME = 1 << 11,
};

ENUM_OPERATORS(eClosureBits, CLOSURE_VOLUME);

static inline eClosureBits extract_closure_mask(const GPUMaterial *gpumat)
{
  eClosureBits closure_bits = eClosureBits(0);
  if (GPU_material_flag_get(gpumat, GPU_MATFLAG_DIFFUSE)) {
    closure_bits |= CLOSURE_DIFFUSE;
  }
  if (GPU_material_flag_get(gpumat, GPU_MATFLAG_TRANSPARENT)) {
    closure_bits |= CLOSURE_TRANSPARENCY;
  }
  if (GPU_material_flag_get(gpumat, GPU_MATFLAG_EMISSION)) {
    closure_bits |= CLOSURE_EMISSION;
  }
  if (GPU_material_flag_get(gpumat, GPU_MATFLAG_GLOSSY)) {
    closure_bits |= CLOSURE_REFLECTION;
  }
  if (GPU_material_flag_get(gpumat, GPU_MATFLAG_SUBSURFACE)) {
    closure_bits |= CLOSURE_SSS;
  }
  if (GPU_material_flag_get(gpumat, GPU_MATFLAG_REFRACT)) {
    closure_bits |= CLOSURE_REFRACTION;
  }
  if (GPU_material_flag_get(gpumat, GPU_MATFLAG_HOLDOUT)) {
    closure_bits |= CLOSURE_HOLDOUT;
  }
  if (GPU_material_flag_get(gpumat, GPU_MATFLAG_AO)) {
    closure_bits |= CLOSURE_AMBIENT_OCCLUSION;
  }
  return closure_bits;
}

struct GBuffer {
  Instance &inst;

  TextureFromPool transmit_color_tx = {"GbufferTransmitColor"};
  TextureFromPool transmit_normal_tx = {"GbufferTransmitNormal"};
  TextureFromPool transmit_data_tx = {"GbufferTransmitData"};

  TextureFromPool reflect_color_tx = {"GbufferReflectionColor"};
  TextureFromPool reflect_normal_tx = {"GbufferReflectionNormal"};

  TextureFromPool radiance_diffuse_tx = {"DiffuseRadiance"};

  /** Raytracing. */
  TextureFromPool ray_data_tx = {"RayData"};
  TextureFromPool ray_radiance_tx = {"RayRadiance"};
  TextureFromPool ray_variance_tx = {"RayVariance"};

  TextureFromPool rpass_emission_tx = {"PassEmission"};
  TextureFromPool rpass_diffuse_light_tx = {"PassDiffuseColor"};
  TextureFromPool rpass_specular_light_tx = {"PassSpecularColor"};
  TextureFromPool rpass_volume_light_tx = {"PassVolumeLight"};

  /* Maximum texture size. Since we use imageLoad/Store instead of framebuffer, we only need to
   * allocate the biggest texture. */
  int2 extent = int2(-1);

  GBuffer(Instance &inst_) : inst(inst_){};

  void begin_sync()
  {
    extent = int2(-1);

    transmit_color_tx.sync();
    transmit_normal_tx.sync();
    transmit_data_tx.sync();
    reflect_color_tx.sync();
    reflect_normal_tx.sync();
    radiance_diffuse_tx.sync();
    ray_data_tx.sync();
    ray_radiance_tx.sync();
    ray_variance_tx.sync();
    rpass_emission_tx.sync();
    rpass_diffuse_light_tx.sync();
    rpass_specular_light_tx.sync();
    rpass_volume_light_tx.sync();
  }

  void view_sync(int2 view_extent)
  {
    extent = math::max(extent, view_extent);
  }

  void acquire(eClosureBits closures_used)
  {
    DrawEngineType *owner = (DrawEngineType *)&inst;

    if (closures_used & (CLOSURE_DIFFUSE | CLOSURE_SSS | CLOSURE_REFRACTION)) {
      transmit_color_tx.acquire(extent, GPU_R11F_G11F_B10F, owner);
    }
    else {
      transmit_color_tx.acquire(int2(1), GPU_R11F_G11F_B10F, owner);
    }

    if (closures_used & (CLOSURE_SSS | CLOSURE_REFRACTION)) {
      transmit_normal_tx.acquire(extent, GPU_RGBA16F, owner);
      transmit_data_tx.acquire(extent, GPU_R11F_G11F_B10F, owner);
    }
    else if (closures_used & CLOSURE_DIFFUSE) {
      transmit_normal_tx.acquire(extent, GPU_RG16F, owner);
      transmit_data_tx.acquire(int2(1), GPU_R11F_G11F_B10F, owner);
    }

    if (closures_used & CLOSURE_REFLECTION) {
      reflect_color_tx.acquire(extent, GPU_R11F_G11F_B10F, owner);
      reflect_normal_tx.acquire(extent, GPU_RGBA16F, owner);
    }
    else {
      reflect_color_tx.acquire(int2(1), GPU_R11F_G11F_B10F, owner);
      reflect_normal_tx.acquire(int2(1), GPU_RGBA16F, owner);
    }

    if (closures_used & CLOSURE_SSS) {
      radiance_diffuse_tx.acquire(extent, GPU_RGBA16F, owner);
    }
    else {
      radiance_diffuse_tx.acquire(int2(1), GPU_RGBA16F, owner);
    }

    if (closures_used & (CLOSURE_DIFFUSE | CLOSURE_REFLECTION | CLOSURE_REFRACTION)) {
      ray_data_tx.acquire(extent, GPU_RGBA16F, owner);
      ray_radiance_tx.acquire(extent, GPU_RGBA16F, owner);
      ray_variance_tx.acquire(extent, GPU_R8, owner);
    }

    if (true) {
      /* Dummies for everything for now. */
      rpass_emission_tx.acquire(int2(1), GPU_RGBA16F, owner);
      rpass_diffuse_light_tx.acquire(int2(1), GPU_RGBA16F, owner);
      rpass_specular_light_tx.acquire(int2(1), GPU_RGBA16F, owner);
      rpass_volume_light_tx.acquire(int2(1), GPU_RGBA16F, owner);
    }
  }

  void release(void)
  {
    transmit_color_tx.release();
    transmit_normal_tx.release();
    transmit_data_tx.release();
    reflect_color_tx.release();
    reflect_normal_tx.release();
    radiance_diffuse_tx.release();
    ray_data_tx.release();
    ray_radiance_tx.release();
    ray_variance_tx.release();
    rpass_emission_tx.release();
    rpass_diffuse_light_tx.release();
    rpass_specular_light_tx.release();
    rpass_volume_light_tx.release();
  }
};

/** \} */

}  // namespace blender::eevee