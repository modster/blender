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
 * The Hierarchical-Z buffer is texture containing a copy of the depth buffer with mipmaps.
 * Each mip contains the maximum depth of each 4 pixels on the upper level.
 * The size of the texture is padded to avoid messing with the mipmap pixels alignments.
 */

#pragma once

#include "DRW_render.h"

#include "eevee_shader_shared.hh"

namespace blender::eevee {

class Instance;

/* -------------------------------------------------------------------- */
/** \name Hierarchical-Z buffer
 * \{ */

class HiZBuffer {
 private:
  Instance &inst_;
  /** Framebuffer use for recursive downsampling. */
  /* TODO(fclem) Remove this and use a compute shader instead. */
  Framebuffer hiz_fb_ = Framebuffer("DepthHiz");
  /** Max mip to downsample to. We ensure the hiz has enough padding to never
   * have to compensate the mipmap alignments. */
  constexpr static int mip_count_ = 6;
  /** TODO/OPTI(fclem): Share it between similar views. */
  Texture hiz_tx_;

 public:
  HiZBuffer(Instance &inst) : inst_(inst){};

  void prepare(GPUTexture *depth_src);
  void update(GPUTexture *depth_src);

  GPUTexture *texture_get(void) const
  {
    return hiz_tx_;
  }

 private:
  static void recursive_downsample(void *thunk, int lvl);
};

/** \} */

/* -------------------------------------------------------------------- */
/** \name Hierarchical-Z buffer Module
 * \{ */

class HiZBufferModule {
  friend HiZBuffer;

 private:
  Instance &inst_;

  HiZDataBuf data_;
  /** Copy input depth to hiz-buffer with border padding. */
  DRWPass *hiz_copy_ps_;
  /** Downsample one mipmap level. */
  DRWPass *hiz_downsample_ps_;
  /** References only. */
  GPUTexture *input_depth_tx_ = nullptr;
  /** Pixel size of the render target during hiz downsampling. */
  vec2 texel_size_;

 public:
  HiZBufferModule(Instance &inst) : inst_(inst){};

  void sync(void);

  const GPUUniformBuf *ubo_get(void) const
  {
    return data_.ubo_get();
  }
};

/** \} */

}  // namespace blender::eevee
