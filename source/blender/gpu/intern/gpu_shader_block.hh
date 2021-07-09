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
 * Copyright 2021, Blender Foundation.
 */

/** \file
 * \ingroup gpu
 */

#pragma once

#include "BLI_sys_types.h"

#include <optional>

#include "GPU_shader_block_types.h"
#include "gpu_backend.hh"
#include "gpu_shader_interface.hh"
#include "gpu_uniform_buffer_private.hh"

#include <array>

namespace blender::gpu {

class ShaderBlockType {
 public:
  constexpr ShaderBlockType(const GPUShaderBlockType type);
  static const ShaderBlockType &get(const GPUShaderBlockType type);

  bool has_all_builtin_uniforms(const ShaderInterface &interface) const;

  GPUShaderBlockType type;
  struct AttributeBinding {
    int location = -1;
    size_t offset = 0;

    bool has_binding() const;
  };

  const AttributeBinding &attribute_binding(const GPUUniformBuiltin builtin_uniform) const
  {
    return m_attribute_bindings[builtin_uniform];
  }

  size_t data_size() const
  {
    return m_data_size;
  }

  const char *defines() const
  {
    return m_defines;
  }

 private:
  const std::array<const AttributeBinding, GPU_NUM_UNIFORMS> &m_attribute_bindings;
  const size_t m_data_size;
  const char *m_defines;
};

class ShaderBlockBuffer {
 public:
  struct Flags {
    bool is_dirty : 1;
  };

  ShaderBlockBuffer(const GPUShaderBlockType type);
  ShaderBlockBuffer(const ShaderBlockBuffer &other) = default;
  ShaderBlockBuffer(ShaderBlockBuffer &&other) = default;

  ~ShaderBlockBuffer();

  const Flags &flags() const
  {
    return m_flags;
  }

  void *data() const
  {
    return m_data;
  };

  const ShaderBlockType &type_info() const
  {
    return m_type_info;
  }

  bool uniform_int(int location, int comp_len, int array_size, const int *data);
  bool uniform_float(int location, int comp_len, int array_size, const float *data);
  void update();
  void bind(int binding);

 private:
  Flags m_flags;
  const ShaderBlockType &m_type_info;
  void *m_data;
  UniformBuf *m_ubo;
};

std::optional<const GPUShaderBlockType> find_smallest_uniform_builtin_struct(
    const ShaderInterface &interface);

}  // namespace blender::gpu
