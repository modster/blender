
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
 * The Original Code is Copyright (C) 2020 Blender Foundation.
 * All rights reserved.
 */

/** \file
 * \ingroup gpu
 *
 * GPU Framebuffer
 * - this is a wrapper for an OpenGL framebuffer object (FBO). in practice
 *   multiple FBO's may be created.
 * - actual FBO creation & config is deferred until GPU_framebuffer_bind or
 *   GPU_framebuffer_check_valid to allow creation & config while another
 *   opengl context is bound (since FBOs are not shared between ogl contexts).
 */

#pragma once

#include "MEM_guardedalloc.h"

#include "BLI_assert.h"

#include "gpu_texture_private.hh"

struct GPUFrameBuffer;

namespace blender {
namespace gpu {

class VKTexture : public Texture {
 public:
  VKTexture(const char *name) : Texture(name){};
  ~VKTexture(){};

  void update_sub(
      int mip, int offset[3], int extent[3], eGPUDataFormat type, const void *data) override{};

  void generate_mipmap(void) override{};
  void copy_to(Texture *dst) override{};
  void clear(eGPUDataFormat format, const void *data) override{};
  void swizzle_set(const char swizzle_mask[4]) override{};
  void mip_range_set(int min, int max) override{};
  void *read(int mip, eGPUDataFormat type) override
  {
    /* NOTE: mip_size_get() won't override any dimension that is equal to 0. */
    int extent[3] = {1, 1, 1};
    this->mip_size_get(mip, extent);

    size_t sample_len = extent[0] * extent[1] * extent[2];
    size_t sample_size = to_bytesize(format_, type);
    size_t texture_size = sample_len * sample_size;
    return MEM_callocN(texture_size, __func__);
  };

  /* TODO(fclem) Legacy. Should be removed at some point. */
  uint gl_bindcode_get(void) const override
  {
    return 0;
  };

 protected:
  bool init_internal(void) override
  {
    return true;
  };
  bool init_internal(GPUVertBuf *vbo) override
  {
    return true;
  };

  MEM_CXX_CLASS_ALLOC_FUNCS("VKTexture")
};

}  // namespace gpu
}  // namespace blender
