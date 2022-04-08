/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#pragma once

#include <cstdint>

#include "BLI_map.hh"
#include "BLI_math_vec_types.hh"
#include "BLI_vector.hh"

#include "GPU_texture.h"

namespace blender::viewport_compositor {

/* A key structure used to identify a texture specification in a texture pool. Defines a hash and
 * an equality operator for use in a hash map. */
class TexturePoolKey {
 public:
  int2 size;
  eGPUTextureFormat format;

  TexturePoolKey(int2 size, eGPUTextureFormat format);
  TexturePoolKey(const GPUTexture *texture);

  uint64_t hash() const;
};

/* A pool of textures that can be used to allocate textures and reused transparently throughout the
 * evaluation of the compositor. This texture pool only pools textures throughout a single
 * evaluation of the compositor and will reset after the evaluation without freeing any textures.
 * Cross-evaluation pooling and freeing of unused textures is the responsibility of the back-end
 * texture pool used by the allocate_texture method. In the case of the viewport compositor engine,
 * this would be the global DRWTexturePool of the draw manager. */
class TexturePool {
 private:
  /* The set of textures in the pool that are available to acquire for each distinct texture
   * specification. */
  Map<TexturePoolKey, Vector<GPUTexture *>> textures_;

 public:
  /* Check if there is an available texture with the given specification in the pool, if such
   * texture exists, return it, otherwise, return a newly allocated texture. Expect the texture to
   * be uncleared and contains garbage data. */
  GPUTexture *acquire(int2 size, eGPUTextureFormat format);

  /* Shorthand for acquire with GPU_RGBA16F format. */
  GPUTexture *acquire_color(int2 size);

  /* Shorthand for acquire with GPU_RGBA16F format. Identical to acquire_color because vector
   * textures are and should internally be stored in RGBA textures. */
  GPUTexture *acquire_vector(int2 size);

  /* Shorthand for acquire with GPU_R16F format. */
  GPUTexture *acquire_float(int2 size);

  /* Put the texture back into the pool, potentially to be acquired later by another user. Expects
   * the texture to be one that was acquired using the same texture pool. */
  void release(GPUTexture *texture);

 private:
  /* Returns a newly allocated texture with the given specification. This method should be
   * implemented by the compositor engine and should use a global texture pool that is persistent
   * across evaluations and capable of freeing unused textures. In the case of the viewport
   * compositor engine, this would be the global DRWTexturePool of the draw manager. */
  virtual GPUTexture *allocate_texture(int2 size, eGPUTextureFormat format) = 0;
};

}  // namespace blender::viewport_compositor
