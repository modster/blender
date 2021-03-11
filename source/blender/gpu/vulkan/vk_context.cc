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

#include "GHOST_C-api.h"

#include "vk_framebuffer.hh"
#include "vk_immediate.hh"
#include "vk_state.hh"

#include "vk_context.hh"

namespace blender::gpu {

/* -------------------------------------------------------------------- */
/** \name Constructor / Destructor
 * \{ */

VKContext::VKContext(void *ghost_window, void *ghost_context)
{
  ghost_window_ = ghost_window;

  if (ghost_window != nullptr) {
    ghost_context = GHOST_GetDrawingContext((GHOST_WindowHandle)ghost_window);
  }

  state_manager = new VKStateManager();
  imm = new VKImmediate();

  GHOST_GetVulkanHandles((GHOST_ContextHandle)ghost_context,
                         &instance_,
                         &physical_device_,
                         &device_,
                         &graphic_queue_familly_);

  /* For offscreen contexts. Default framebuffer is empty. */
  back_left = new VKFrameBuffer("back_left");
}

VKContext::~VKContext()
{
}

void VKContext::activate()
{
  if (ghost_window_) {
    VkImage image; /* TODO will be used for reading later... */
    VkFramebuffer framebuffer;
    VkCommandBuffer command_buffer;
    VkRenderPass render_pass;
    VkExtent2D extent;
    uint32_t fb_id;

    GHOST_GetVulkanBackbuffer((GHOST_WindowHandle)ghost_window_,
                              &image,
                              &framebuffer,
                              &command_buffer,
                              &render_pass,
                              &extent,
                              &fb_id);

    if (fb_id != fb_id_) {
      /* Recreate the gpu::VKFrameBuffer wrapper after every swap. */
      delete back_left;
    }

    back_left = new VKFrameBuffer("back_left", framebuffer, command_buffer, render_pass, extent);
    active_fb = back_left;
  }

  immActivate();
}

void VKContext::deactivate()
{
  immDeactivate();
}

/** \} */

}  // namespace blender::gpu