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
 */

#include "BLI_vector.hh"

#include "vk_context.hh"
#include "vk_texture.hh"

#include "vk_framebuffer.hh"

namespace blender::gpu {

/* -------------------------------------------------------------------- */
/** \name Creation & Deletion
 * \{ */

VKFrameBuffer::VKFrameBuffer(const char *name) : FrameBuffer(name)
{
  /* Just-In-Time init. See #VKFrameBuffer::init(). */
  immutable_ = false;
}

VKFrameBuffer::VKFrameBuffer(const char *name,
                             VkFramebuffer framebuffer,
                             VkCommandBuffer command_buffer,
                             VkRenderPass render_pass,
                             VkExtent2D extent)
    : FrameBuffer(name)
{
  immutable_ = true;
  /* Never update an internal frame-buffer. */
  dirty_attachments_ = false;
  width_ = extent.width;
  height_ = extent.height;
  vk_fb_ = framebuffer;

  viewport_[0] = scissor_[0] = 0;
  viewport_[1] = scissor_[1] = 0;
  viewport_[2] = scissor_[2] = width_;
  viewport_[3] = scissor_[3] = height_;
}

VKFrameBuffer::~VKFrameBuffer()
{
  if (!immutable_ && vk_fb_ != VK_NULL_HANDLE) {
    vkDestroyFramebuffer(vk_device_, vk_fb_, NULL);
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Config
 * \{ */

/** \} */

}  // namespace blender::gpu