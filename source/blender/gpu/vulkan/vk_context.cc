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

  vkGetDeviceQueue(device_, graphic_queue_familly_, 0, &graphic_queue_);

  {
    VkCommandPoolCreateInfo info = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    info.queueFamilyIndex = graphic_queue_familly_;
    vkCreateCommandPool(device_, &info, nullptr, &graphic_cmd_pool_);
  }

  /* For offscreen contexts. Default framebuffer is empty. */
  back_left = new VKFrameBuffer("back_left");

  {
    VmaAllocatorCreateInfo info = {};
    /* TODO use same vulkan version as GHOST. */
    info.vulkanApiVersion = VK_API_VERSION_1_0;
    info.physicalDevice = physical_device_;
    info.device = device_;
    info.instance = instance_;
    vmaCreateAllocator(&info, &mem_allocator_);
  }
}

VKContext::~VKContext()
{
  vkDestroyCommandPool(device_, graphic_cmd_pool_, nullptr);
  vmaDestroyAllocator(mem_allocator_);
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

/* -------------------------------------------------------------------- */
/** \name Command buffers
 * \{ */

VkCommandBuffer VKContext::single_use_command_buffer_begin(void)
{
  VkCommandBuffer cmd_buf;
  {
    VkCommandBufferAllocateInfo info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    info.commandPool = graphic_cmd_pool_;
    info.commandBufferCount = 1;
    vkAllocateCommandBuffers(device_, &info, &cmd_buf);
  }
  {
    VkCommandBufferBeginInfo info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd_buf, &info);
  }
  return cmd_buf;
}

void VKContext::single_use_command_buffer_end(VkCommandBuffer cmd_buf)
{
  vkEndCommandBuffer(cmd_buf);

  this->submit_and_wait(cmd_buf);

  vkFreeCommandBuffers(device_, graphic_cmd_pool_, 1, &cmd_buf);
}

void VKContext::submit_and_wait(VkCommandBuffer cmd_buf)
{
  {
    VkSubmitInfo info = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    info.commandBufferCount = 1;
    info.pCommandBuffers = &cmd_buf;
    vkQueueSubmit(graphic_queue_, 1, &info, VK_NULL_HANDLE);
  }
  vkQueueWaitIdle(graphic_queue_);
}

/** \} */

}  // namespace blender::gpu