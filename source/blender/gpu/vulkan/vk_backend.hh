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
 * Copyright 2020, Blender Foundation.
 * All rights reserved.
 */

/** \file
 * \ingroup gpu
 */

#pragma once

#include "gpu_backend.hh"

#include "vk_batch.hh"
#include "vk_context.hh"
#include "vk_drawlist.hh"
#include "vk_framebuffer.hh"
#include "vk_index_buffer.hh"
#include "vk_query.hh"
#include "vk_shader.hh"
#include "vk_texture.hh"
#include "vk_uniform_buffer.hh"
#include "vk_vertex_buffer.hh"

namespace blender {
namespace gpu {

class VKBackend : public GPUBackend {
 public:
  VKBackend()
  {
    /* platform_init needs to go first. */
    VKBackend::platform_init();

    VKBackend::capabilities_init();
  }
  ~VKBackend()
  {
    VKBackend::platform_exit();
  }

  static VKBackend *get(void)
  {
    return static_cast<VKBackend *>(GPUBackend::get());
  };

  void samplers_update(void) override{};

  Context *context_alloc(void *ghost_window, void *ghost_context) override
  {
    return new VKContext(ghost_window, ghost_context);
  };

  Batch *batch_alloc(void) override
  {
    return new VKBatch();
  };

  DrawList *drawlist_alloc(int list_length) override
  {
    return new VKDrawList(list_length);
  };

  FrameBuffer *framebuffer_alloc(const char *name) override
  {
    return new VKFrameBuffer(name);
  };

  IndexBuf *indexbuf_alloc(void) override
  {
    return new VKIndexBuf();
  };

  QueryPool *querypool_alloc(void) override
  {
    return new VKQueryPool();
  };

  Shader *shader_alloc(const char *name) override
  {
    return new VKShader(name);
  };

  Texture *texture_alloc(const char *name) override
  {
    return new VKTexture(name);
  };

  UniformBuf *uniformbuf_alloc(int size, const char *name) override
  {
    return new VKUniformBuf(size, name);
  };

  VertBuf *vertbuf_alloc(void) override
  {
    return new VKVertBuf();
  };

 private:
  static void platform_init(void);
  static void platform_exit(void);

  static void capabilities_init(void);
};

}  // namespace gpu
}  // namespace blender
