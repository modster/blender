/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2021 Blender Foundation.
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
  /** Max mip to downsample to. We ensure the hiz has enough padding to never
   * have to compensate the mipmap alignments. */
  constexpr static int mip_count_ = HIZ_MIP_COUNT;
  /* Kernel size is bigger than the local group size because we simply copy the level 0. */
  constexpr static int kernel_size_ = HIZ_GROUP_SIZE << 1;
  /** Aggregate of all views extent. The hiz is large enough to fit any view and is shared. */
  int2 extent_;
  /** The texture containing the hiz mip chain. */
  Texture hiz_tx_ = {"hiz_tx_"};
  /** Update dispatch size from the given depth buffer size. */
  int3 dispatch_size_;
  /** Single pass recursive downsample. */
  DRWPass *hiz_update_ps_;
  /** References only. */
  GPUTexture *input_depth_tx_ = nullptr;

 public:
  HiZBuffer(Instance &inst) : inst_(inst){};

  void begin_sync();
  void view_sync(int2 extent);
  void end_sync();

  void prepare(GPUTexture *depth_src_tx);

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

 public:
  HiZBufferModule(Instance &inst) : inst_(inst){};

  const GPUUniformBuf *ubo_get(void) const
  {
    return data_;
  }
};

/** \} */

}  // namespace blender::eevee
