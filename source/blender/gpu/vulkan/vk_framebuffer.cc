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
  fbo_id_ = 0;
}

VKFrameBuffer::VKFrameBuffer(const char *name,
                             VkFramebuffer framebuffer,
                             VkCommandBuffer command_buffer,
                             VkRenderPass render_pass,
                             VkExtent2D extent)
    : FrameBuffer(name)
{
  device_ = VKContext::get()->device_get();
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

void VKFrameBuffer::attachment_descriptions_get(Vector<VkAttachmentDescription> descriptions,
                                                Vector<VkAttachmentReference> references,
                                                Vector<VkImageView> views)
{
  int size[3] = {1, 1, 1};
  bool maybe_layered = true;
  VKTexture *vk_tex = NULL;

  for (GPUAttachmentType type = GPU_FB_MAX_ATTACHMENT - 1; type >= 0; --type) {
    GPUAttachment &attach = attachments_[type];
    VkAttachmentReference &reference = references[to_vk_slot(type)];
    /* TODO: More optimal layout. */
    reference.layout = VK_IMAGE_LAYOUT_GENERAL;
    reference.attachment = (attach.tex) ? descriptions.size() : VK_ATTACHMENT_UNUSED;

    if (attach.tex == NULL) {
      continue;
    }

    vk_tex = static_cast<VKTexture *>(unwrap(attach.tex));
    vk_tex->mip_size_get(attach.mip, size);

    VkAttachmentDescription description = {};
    description.format = vk_tex->vk_format_;
    description.samples = VK_SAMPLE_COUNT_1_BIT;
    description.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    description.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    description.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    description.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    description.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    description.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    descriptions.append(description);

    VkImageView view;
    if (attach.layer > -1) {
      view = vk_tex->get_image_layer_view(attach.mip, attach.layer);
      maybe_layered = false;
    }
    else {
      view = vk_tex->get_image_view(attach.mip);
    }
    views.append(view);

    /* We found one depth buffer type. Stop here, otherwise we would
     * override it by setting GPU_FB_DEPTH_ATTACHMENT */
    /* TODO(fclem) remove this by getting the attachement type from the texture type. */
    if (type == GPU_FB_DEPTH_STENCIL_ATTACHMENT) {
      break;
    }
  }

  width_ = size[0];
  height_ = size[1];
  depth_ = maybe_layered ? size[2] : 1;
}

void VKFrameBuffer::configure(void)
{
  if (vk_fb_ != VK_NULL_HANDLE) {
    vkDestroyFramebuffer(vk_device_, vk_fb_, NULL);
  }

  sizeof(VkAttachmentDescription);

  Vector<VkAttachmentDescription, VK_MAX_ATTACHMENT> descriptions;
  Vector<VkAttachmentReference, VK_MAX_ATTACHMENT> references(VK_MAX_ATTACHMENT);
  Vector<VkImageView, VK_MAX_ATTACHMENT> views;

  this->attachment_descriptions_get(descriptions, references, views);

  {
    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = GPU_FB_MAX_COLOR_ATTACHMENT;
    subpass.pColorAttachments = &references[1];
    subpass.pDepthStencilAttachment = &references[0];

    VkRenderPassCreateInfo info = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    info.attachmentCount = descriptions.size();
    info.pAttachments = descriptions.data();
    info.subpassCount = 1;
    info.pSubpasses = &subpass;

    vkCreateRenderPass(vk_device_, &info, NULL, render_pass_);
  }

  {
    VkFramebufferCreateInfo info = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    info.renderPass = render_pass_;
    info.attachmentCount = views.size();
    info.pAttachments = views.data();
    info.width = width_;
    info.height = height_;
    info.layers = depth_;

    vkCreateFramebuffer(vk_device_, &info, NULL, &vk_fb_);
  }

  dirty_attachments_ = false;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Config
 * \{ */

/** \} */

}  // namespace blender::gpu