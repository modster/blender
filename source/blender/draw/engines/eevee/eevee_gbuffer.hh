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
  CLOSURE_VOLUME = 1 << 4,
  CLOSURE_EMISSION = 1 << 5,
  CLOSURE_TRANSPARENCY = 1 << 6,
};

struct GBuffer {
  TextureFromPool transmit_color_tx = {"GbufferTransmitColor"};
  TextureFromPool transmit_normal_tx = {"GbufferTransmitNormal"};
  TextureFromPool transmit_data_tx = {"GbufferTransmitData"};
  TextureFromPool reflect_color_tx = {"GbufferReflectionColor"};
  TextureFromPool reflect_normal_tx = {"GbufferReflectionNormal"};
  TextureFromPool volume_tx = {"GbufferVolume"};
  TextureFromPool emission_tx = {"GbufferEmission"};
  TextureFromPool transparency_tx = {"GbufferTransparency"};

  Framebuffer gbuffer_fb = {"Gbuffer"};
  Framebuffer volume_fb = {"VolumeHeterogeneous"};

  TextureFromPool holdout_tx = {"HoldoutRadiance"};
  TextureFromPool diffuse_tx = {"DiffuseRadiance"};

  Framebuffer radiance_fb = {"Radiance"};
  Framebuffer radiance_clear_fb = {"RadianceClear"};

  Framebuffer holdout_fb = {"Holdout"};

  TextureFromPool depth_behind_tx = {"DepthBehind"};

  Framebuffer depth_behind_fb = {"DepthCopy"};

  /** Raytracing. */
  TextureFromPool ray_data_tx = {"RayData"};
  TextureFromPool ray_radiance_tx = {"RayRadiance"};
  TextureFromPool ray_variance_tx = {"RayVariance"};
  Framebuffer ray_data_fb = {"RayData"};
  Framebuffer ray_denoise_fb = {"RayDenoise"};

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
    transmit_color_tx.sync();
    transmit_normal_tx.sync();
    transmit_data_tx.sync();
    reflect_color_tx.sync();
    reflect_normal_tx.sync();
    volume_tx.sync();
    emission_tx.sync();
    transparency_tx.sync();
    holdout_tx.sync();
    diffuse_tx.sync();
    depth_behind_tx.sync();
    ray_data_tx.sync();
    ray_radiance_tx.sync();
    ray_variance_tx.sync();
  }

  void prepare(eClosureBits closures_used)
  {
    int2 extent = {GPU_texture_width(depth_tx), GPU_texture_height(depth_tx)};

    /* TODO Reuse for different config. */
    if (closures_used & (CLOSURE_DIFFUSE | CLOSURE_SSS | CLOSURE_REFRACTION)) {
      transmit_color_tx.acquire(extent, GPU_R11F_G11F_B10F, owner);
    }
    if (closures_used & (CLOSURE_SSS | CLOSURE_REFRACTION)) {
      transmit_normal_tx.acquire(extent, GPU_RGBA16F, owner);
      transmit_data_tx.acquire(extent, GPU_R11F_G11F_B10F, owner);
    }
    else if (closures_used & CLOSURE_DIFFUSE) {
      transmit_normal_tx.acquire(extent, GPU_RG16F, owner);
    }
    if (closures_used & CLOSURE_SSS) {
      diffuse_tx.acquire(extent, GPU_RGBA16F, owner);
    }

    if (closures_used & CLOSURE_REFLECTION) {
      reflect_color_tx.acquire(extent, GPU_R11F_G11F_B10F, owner);
      reflect_normal_tx.acquire(extent, GPU_RGBA16F, owner);
    }

    if (closures_used & CLOSURE_VOLUME) {
      /* TODO(fclem): This is killing performance.
       * Idea: use interleaved data pattern to fill only a 32bpp buffer. */
      volume_tx.acquire(extent, GPU_RGBA32UI, owner);
    }

    if (closures_used & CLOSURE_EMISSION) {
      emission_tx.acquire(extent, GPU_R11F_G11F_B10F, owner);
    }

    if (closures_used & CLOSURE_TRANSPARENCY) {
      /* TODO(fclem): Speedup by using Dithered holdout and GPU_RGB10_A2. */
      transparency_tx.acquire(extent, GPU_RGBA16, owner);
    }

    if (closures_used & (CLOSURE_DIFFUSE | CLOSURE_REFLECTION | CLOSURE_REFRACTION)) {
      ray_data_tx.acquire(extent, GPU_RGBA16F, owner);
      ray_radiance_tx.acquire(extent, GPU_RGBA16F, owner);
      ray_variance_tx.acquire(extent, GPU_R8, owner);
    }

    holdout_tx.acquire(extent, GPU_R11F_G11F_B10F, owner);
    depth_behind_tx.acquire(extent, GPU_DEPTH24_STENCIL8, owner);

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
  }

  void bind(void)
  {
    GPU_framebuffer_bind(gbuffer_fb);
    GPU_framebuffer_clear_stencil(gbuffer_fb, 0x0);
  }

  void bind_radiance(void)
  {
    /* Layer attachement also works with cubemap. */
    radiance_fb.ensure(GPU_ATTACHMENT_TEXTURE_LAYER(depth_tx, layer),
                       GPU_ATTACHMENT_TEXTURE(combined_tx),
                       GPU_ATTACHMENT_TEXTURE(diffuse_tx));
    GPU_framebuffer_bind(radiance_fb);
  }

  void bind_volume(void)
  {
    /* Layer attachement also works with cubemap. */
    volume_fb.ensure(GPU_ATTACHMENT_TEXTURE_LAYER(depth_tx, layer),
                     GPU_ATTACHMENT_TEXTURE(volume_tx),
                     GPU_ATTACHMENT_TEXTURE(transparency_tx));
    GPU_framebuffer_bind(volume_fb);
  }

  void bind_tracing(void)
  {
    /* Layer attachement also works with cubemap. */
    /* Attach depth_stencil buffer to only trace the surfaces that need it. */
    ray_data_fb.ensure(GPU_ATTACHMENT_TEXTURE_LAYER(depth_tx, layer),
                       GPU_ATTACHMENT_TEXTURE(ray_data_tx),
                       GPU_ATTACHMENT_TEXTURE(ray_radiance_tx));
    GPU_framebuffer_bind(ray_data_fb);

    float color[4] = {0.0f};
    GPU_framebuffer_clear_color(ray_data_fb, color);
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

  void clear_radiance(void)
  {
    radiance_clear_fb.ensure(GPU_ATTACHMENT_NONE, GPU_ATTACHMENT_TEXTURE(diffuse_tx));
    GPU_framebuffer_bind(radiance_clear_fb);

    float color[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    GPU_framebuffer_clear_color(radiance_clear_fb, color);
  }

  void render_end(void)
  {
    transmit_color_tx.release();
    transmit_normal_tx.release();
    transmit_data_tx.release();
    reflect_color_tx.release();
    reflect_normal_tx.release();
    volume_tx.release();
    emission_tx.release();
    transparency_tx.release();
    holdout_tx.release();
    diffuse_tx.release();
    depth_behind_tx.release();
    ray_data_tx.release();
    ray_radiance_tx.release();
    ray_variance_tx.release();
  }
};

/** \} */

}  // namespace blender::eevee