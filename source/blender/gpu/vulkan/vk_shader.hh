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

#pragma once

#include "MEM_guardedalloc.h"

#include <vulkan/vulkan.h>

#include "gpu_shader_private.hh"

/* TODO move this deps to the .cc file. */
#include "vk_context.hh"
#include "vk_shader_interface.hh"

namespace blender {
namespace gpu {

enum class VKShaderStageType {
  VertexShader,
  GeometryShader,
  FragmentShader,
  ComputeShader,
};

std::ostream &operator<<(std::ostream &os, const VKShaderStageType &stage);

/**
 * Implementation of shader compilation and uniforms handling using OpenGL.
 **/
class VKShader : public Shader {
 private:
  VKContext *context_ = nullptr;

  VkShaderModule vertex_shader_ = VK_NULL_HANDLE;
  VkShaderModule geometry_shader_ = VK_NULL_HANDLE;
  VkShaderModule fragment_shader_ = VK_NULL_HANDLE;
  VkShaderModule compute_shader_ = VK_NULL_HANDLE;

  /** True if any shader failed to compile. */
  bool compilation_failed_ = false;

 public:
  VKShader(const char *name);
  ~VKShader();

  void vertex_shader_from_glsl(MutableSpan<const char *> sources) override;
  void geometry_shader_from_glsl(MutableSpan<const char *> sources) override;
  void fragment_shader_from_glsl(MutableSpan<const char *> sources) override;
  /* Return true on success. */
  bool finalize(void);

  void transform_feedback_names_set(Span<const char *> name_list,
                                    const eGPUShaderTFBType geom_type) override{};
  bool transform_feedback_enable(GPUVertBuf *buf) override
  {
    return true;
  };
  void transform_feedback_disable(void) override{};

  void bind(void) override{};
  void unbind(void) override{};

  void uniform_float(int location, int comp_len, int array_size, const float *data) override{};
  void uniform_int(int location, int comp_len, int array_size, const int *data) override{};

  void vertformat_from_shader(GPUVertFormat *format) const override{};

  int program_handle_get() const override
  {
    return 0;
  }

 private:
  std::unique_ptr<std::vector<uint32_t>> compile_source(MutableSpan<const char *> sources,
                                                        VKShaderStageType stage);
  VkShaderModule create_shader_module(MutableSpan<const char *> sources, VKShaderStageType stage);

  MEM_CXX_CLASS_ALLOC_FUNCS("VKShader");
};

}  // namespace gpu
}  // namespace blender
