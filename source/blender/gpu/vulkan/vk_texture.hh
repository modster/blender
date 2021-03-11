
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

#include <vulkan/vulkan.h>

#include "BLI_assert.h"
#include "BLI_vector.hh"

#include "gpu_texture_private.hh"

struct GPUFrameBuffer;

namespace blender {
namespace gpu {

class VKTexture : public Texture {
 protected:
  /* Vulkan context who created the object. */
  VKContext *context_ = nullptr;
  /* Vulkan object handle. */
  VkImage vk_image_ = VK_NULL_HANDLE;
  /* GPU Memory allocated by this object. */
  VmaAllocation vk_allocation_ = VK_NULL_HANDLE;
  /* Vulkan format used to initialize the texture. */
  VkFormat vk_format_ = VK_FORMAT_MAX_ENUM;
  /* Image views for each mipmap and each layer. */
  Vector<VkImageView> views_;
  /* Swizzle state. */
  VkComponentMapping vk_swizzle_ = {VK_COMPONENT_SWIZZLE_IDENTITY,
                                    VK_COMPONENT_SWIZZLE_IDENTITY,
                                    VK_COMPONENT_SWIZZLE_IDENTITY,
                                    VK_COMPONENT_SWIZZLE_IDENTITY};

 public:
  VKTexture(const char *name);
  ~VKTexture();

  void update_sub(
      int mip, int offset[3], int extent[3], eGPUDataFormat type, const void *data) override;

  void generate_mipmap(void) override{};
  void copy_to(Texture *dst) override{};
  void clear(eGPUDataFormat format, const void *data) override{};
  void swizzle_set(const char swizzle_mask[4]) override;
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

  /* Vulkan specific functions. */
  VkImageView vk_image_view_get(int mip);
  VkImageView vk_image_view_get(int mip, int layer);

  VkFormat vk_format_get(void) const
  {
    return this->vk_format_;
  }

 protected:
  bool init_internal(void) override;
  bool init_internal(GPUVertBuf *vbo) override
  {
    return true;
  };

  VkImageView create_image_view(int mip, int layer);

  MEM_CXX_CLASS_ALLOC_FUNCS("VKTexture")
};

inline VkImageType to_vk_image_type(eGPUTextureType type)
{
  switch (type & (GPU_TEXTURE_1D | GPU_TEXTURE_2D | GPU_TEXTURE_3D)) {
    case GPU_TEXTURE_1D:
      return VK_IMAGE_TYPE_1D;
    case GPU_TEXTURE_2D:
      return VK_IMAGE_TYPE_2D;
    case GPU_TEXTURE_3D:
      return VK_IMAGE_TYPE_3D;
    default:
      BLI_assert(!"Wrong enum!");
      return VK_IMAGE_TYPE_MAX_ENUM;
  }
}

inline VkImageViewType to_vk_image_view_type(eGPUTextureType type)
{
  switch (type) {
    case GPU_TEXTURE_1D:
      return VK_IMAGE_VIEW_TYPE_1D;
    case GPU_TEXTURE_2D:
      return VK_IMAGE_VIEW_TYPE_2D;
    case GPU_TEXTURE_3D:
      return VK_IMAGE_VIEW_TYPE_3D;
    case GPU_TEXTURE_CUBE:
      return VK_IMAGE_VIEW_TYPE_CUBE;
    case GPU_TEXTURE_1D_ARRAY:
      return VK_IMAGE_VIEW_TYPE_1D_ARRAY;
    case GPU_TEXTURE_2D_ARRAY:
      return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    case GPU_TEXTURE_CUBE_ARRAY:
      return VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
    default:
      BLI_assert(!"Wrong enum!");
      return VK_IMAGE_VIEW_TYPE_MAX_ENUM;
  }
}

inline VkImageAspectFlags to_vk(eGPUTextureFormatFlag flag)
{
  if (flag & GPU_FORMAT_DEPTH) {
    return VK_IMAGE_ASPECT_DEPTH_BIT;
  }
  return VK_IMAGE_ASPECT_COLOR_BIT;
}

inline VkComponentSwizzle swizzle_to_vk(const char swizzle)
{
  switch (swizzle) {
    default:
    case 'x':
    case 'r':
      return VK_COMPONENT_SWIZZLE_R;
    case 'y':
    case 'g':
      return VK_COMPONENT_SWIZZLE_G;
    case 'z':
    case 'b':
      return VK_COMPONENT_SWIZZLE_B;
    case 'w':
    case 'a':
      return VK_COMPONENT_SWIZZLE_A;
    case '0':
      return VK_COMPONENT_SWIZZLE_ZERO;
    case '1':
      return VK_COMPONENT_SWIZZLE_ONE;
  }
}

inline VkFormat to_vk(eGPUTextureFormat format)
{
  switch (format) {
    /* Formats texture & renderbuffer */
    case GPU_RGBA8UI:
      return VK_FORMAT_R8G8B8A8_UINT;
    case GPU_RGBA8I:
      return VK_FORMAT_R8G8B8A8_SINT;
    case GPU_RGBA8:
      return VK_FORMAT_R8G8B8A8_UNORM;
    case GPU_RGBA32UI:
      return VK_FORMAT_R32G32B32A32_UINT;
    case GPU_RGBA32I:
      return VK_FORMAT_R32G32B32A32_SINT;
    case GPU_RGBA32F:
      return VK_FORMAT_R32G32B32A32_SFLOAT;
    case GPU_RGBA16UI:
      return VK_FORMAT_R16G16B16A16_UINT;
    case GPU_RGBA16I:
      return VK_FORMAT_R16G16B16A16_SINT;
    case GPU_RGBA16F:
      return VK_FORMAT_R16G16B16A16_SFLOAT;
    case GPU_RGBA16:
      return VK_FORMAT_R16G16B16A16_UNORM;
    case GPU_RG8UI:
      return VK_FORMAT_R8G8_UINT;
    case GPU_RG8I:
      return VK_FORMAT_R8G8_SINT;
    case GPU_RG8:
      return VK_FORMAT_R8G8_UNORM;
    case GPU_RG32UI:
      return VK_FORMAT_R32G32_UINT;
    case GPU_RG32I:
      return VK_FORMAT_R32G32_SINT;
    case GPU_RG32F:
      return VK_FORMAT_R32G32_SFLOAT;
    case GPU_RG16UI:
      return VK_FORMAT_R16G16_UINT;
    case GPU_RG16I:
      return VK_FORMAT_R16G16_SINT;
    case GPU_RG16F:
      return VK_FORMAT_R16G16_SFLOAT;
    case GPU_RG16:
      return VK_FORMAT_R16G16_UNORM;
    case GPU_R8UI:
      return VK_FORMAT_R8_UINT;
    case GPU_R8I:
      return VK_FORMAT_R8_SINT;
    case GPU_R8:
      return VK_FORMAT_R8_UNORM;
    case GPU_R32UI:
      return VK_FORMAT_R32_UINT;
    case GPU_R32I:
      return VK_FORMAT_R32_SINT;
    case GPU_R32F:
      return VK_FORMAT_R32_SFLOAT;
    case GPU_R16UI:
      return VK_FORMAT_R16_UINT;
    case GPU_R16I:
      return VK_FORMAT_R16_SINT;
    case GPU_R16F:
      return VK_FORMAT_R16_SFLOAT;
    case GPU_R16:
      return VK_FORMAT_R16_UNORM;
    /* Special formats texture & renderbuffer */
    case GPU_RGB10_A2:
      return VK_FORMAT_A2R10G10B10_UNORM_PACK32;
    case GPU_R11F_G11F_B10F:
      return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    case GPU_DEPTH32F_STENCIL8:
      return VK_FORMAT_D32_SFLOAT_S8_UINT;
    case GPU_DEPTH24_STENCIL8:
      return VK_FORMAT_D24_UNORM_S8_UINT;
    case GPU_SRGB8_A8:
      return VK_FORMAT_A8B8G8R8_SRGB_PACK32;
    /* Texture only format */
    case GPU_RGB16F:
      return VK_FORMAT_R16G16B16_SFLOAT;
    /* Special formats texture only */
    case GPU_SRGB8_A8_DXT1:
      return VK_FORMAT_BC1_RGBA_SRGB_BLOCK;
    case GPU_SRGB8_A8_DXT3:
      return VK_FORMAT_BC2_SRGB_BLOCK;
    case GPU_SRGB8_A8_DXT5:
      return VK_FORMAT_BC3_SRGB_BLOCK;
    case GPU_RGBA8_DXT1:
      return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
    case GPU_RGBA8_DXT3:
      return VK_FORMAT_BC2_UNORM_BLOCK;
    case GPU_RGBA8_DXT5:
      return VK_FORMAT_BC3_UNORM_BLOCK;
    /* Depth Formats */
    case GPU_DEPTH_COMPONENT32F:
      return VK_FORMAT_D32_SFLOAT;
    case GPU_DEPTH_COMPONENT24:
      return VK_FORMAT_X8_D24_UNORM_PACK32;
    case GPU_DEPTH_COMPONENT16:
      return VK_FORMAT_D16_UNORM;
  }
}

}  // namespace gpu
}  // namespace blender
