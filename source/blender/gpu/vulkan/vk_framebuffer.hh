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
 * Encapsulation of Framebuffer states (attached textures, viewport, scissors).
 */

#pragma once

#include "MEM_guardedalloc.h"

#include <vulkan/vulkan.h>

#include "gpu_framebuffer_private.hh"

namespace blender::gpu {

class VKContext;

#define VK_MAX_ATTACHMENT (GPU_FB_MAX_COLOR_ATTACHMENT + 1)

/**
 * Implementation of FrameBuffer object using Vulkan.
 **/
class VKFrameBuffer : public FrameBuffer {
 private:
  /* Vulkan object handle. */
  VkFramebuffer vk_fb_ = VK_NULL_HANDLE;
  /* Vulkan device who created the handle. */
  VkDevice vk_device_ = VK_NULL_HANDLE;
  /* Base render pass used for framebuffer creation. */
  VkRenderPass render_pass_ = VK_NULL_HANDLE;
  /* Number of layers if the attachments are layered textures. */
  int depth_ = 1;
  /** Internal frame-buffers are immutable. */
  bool immutable_;

 public:
  /**
   * Create a conventional framebuffer to attach texture to.
   **/
  VKFrameBuffer(const char *name);

  /**
   * Special frame-buffer encapsulating internal window frame-buffer.
   * This just act as a wrapper, the actual allocations are done by GHOST_ContextVK.
   **/
  VKFrameBuffer(const char *name,
                VkFramebuffer framebuffer,
                VkCommandBuffer command_buffer,
                VkRenderPass render_pass,
                VkExtent2D extent);

  ~VKFrameBuffer();

  void bind(bool enabled_srgb) override{};

  bool check(char err_out[256]) override
  {
    return true;
  };

  void clear(eGPUFrameBufferBits buffers,
             const float clear_col[4],
             float clear_depth,
             uint clear_stencil) override{};
  void clear_multi(const float (*clear_cols)[4]) override{};
  void clear_attachment(GPUAttachmentType type,
                        eGPUDataFormat data_format,
                        const void *clear_value) override{};

  void read(eGPUFrameBufferBits planes,
            eGPUDataFormat format,
            const int area[4],
            int channel_len,
            int slot,
            void *r_data) override{};

  void blit_to(eGPUFrameBufferBits planes,
               int src_slot,
               FrameBuffer *dst,
               int dst_slot,
               int dst_offset_x,
               int dst_offset_y) override{};

 private:
  MEM_CXX_CLASS_ALLOC_FUNCS("VKFrameBuffer");
};

/* -------------------------------------------------------------------- */
/** \name Enums Conversion
 * \{ */

/** \} */

}  // namespace blender::gpu
