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
 * The Original Code is Copyright (C) 2021 Blender Foundation.
 * All rights reserved.
 */

/** \file
 * \ingroup gpu
 */

#include "vk_context.hh"

#include "vk_texture.hh"

namespace blender::gpu {

VKTexture::VKTexture(const char *name) : Texture(name)
{
  context_ = VKContext::get();
}

VKTexture::~VKTexture(void)
{
  VkDevice device = context_->device_get();
  for (VkImageView view : views_) {
    if (view != VK_NULL_HANDLE) {
      vkDestroyImageView(device, view, nullptr);
    }
  }
  if (vk_image_ != VK_NULL_HANDLE) {
    vkDestroyImage(device, vk_image_, nullptr);
  }
}

bool VKTexture::init_internal(void)
{
  VmaAllocator mem_allocator = context_->mem_allocator_get();
  {
    /* Usage very rough for now to support same level of interaction as OpenGL in a simple way.
     * TODO(fclem) improve this. */
    VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                              VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    usage |= (format_flag_ & GPU_FORMAT_DEPTH) ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT :
                                                 VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    VkImageCreateInfo info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = to_vk_image_type(type_);
    info.extent.width = static_cast<uint32_t>(w_);
    info.extent.height = static_cast<uint32_t>(h_);
    info.extent.depth = static_cast<uint32_t>(d_);
    /* TODO(fclem) mipmap support. */
    info.mipLevels = 1;
    info.arrayLayers = this->layer_count();
    info.format = to_vk(format_);
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    /* TODO(fclem) improve this. */
    info.tiling = VK_IMAGE_TILING_LINEAR;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    info.usage = usage;
    info.flags = 0;

    views_.resize(info.mipLevels * (info.arrayLayers + 1));

    VmaAllocationCreateInfo alloc_info = {};
    alloc_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    alloc_info.preferredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    alloc_info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    alloc_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    vmaCreateImage(mem_allocator, &info, &alloc_info, &vk_image_, &vk_allocation_, nullptr);
  }

  return true;
}

void VKTexture::update_sub(
    int mip, int offset[3], int extent[3], eGPUDataFormat type, const void *data)
{
}

VkImageView VKTexture::create_image_view(int mip, int layer)
{
  VkDevice device = context_->device_get();

  VkImageSubresourceRange range;
  range.aspectMask = to_vk(format_flag_);
  range.baseMipLevel = (mip > -1) ? mip : 0;
  range.baseArrayLayer = (layer > -1) ? layer : 0;
  range.levelCount = (mip > -1) ? 1 : VK_REMAINING_MIP_LEVELS;
  range.layerCount = (layer > -1) ? 1 : VK_REMAINING_ARRAY_LAYERS;

  /* TODO correct swizzling. */
  VkComponentMapping components;
  components.r = VK_COMPONENT_SWIZZLE_R;
  components.g = VK_COMPONENT_SWIZZLE_G;
  components.b = VK_COMPONENT_SWIZZLE_B;
  components.a = VK_COMPONENT_SWIZZLE_A;

  VkImageViewCreateInfo info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  info.flags = 0;
  info.image = vk_image_;
  info.viewType = to_vk_image_view_type(type_);
  info.format = vk_format_;
  info.components = components;
  info.subresourceRange = range;

  VkImageView view;
  vkCreateImageView(device, &info, nullptr, &view);
  return view;
}

VkImageView VKTexture::vk_image_view_get(int mip)
{
  return this->vk_image_view_get(mip, -1);
}

VkImageView VKTexture::vk_image_view_get(int mip, int layer)
{
  int view_id = mip * (layer_count() + 1) + layer + 1;
  VkImageView view = views_[view_id];
  if (view == VK_NULL_HANDLE) {
    views_[view_id] = view = this->create_image_view(mip, layer);
  }
  return view;
}

}  // namespace blender::gpu
