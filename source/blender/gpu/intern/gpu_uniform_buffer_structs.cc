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

#include "gpu_uniform_buffer_private.hh"

#include "MEM_guardedalloc.h"

#include "GPU_shader.h"
#include "GPU_uniform_buffer_types.h"
#include "gpu_shader_interface.hh"

namespace blender::gpu {

/* -------------------------------------------------------------------- */
/** \name Attribute bindings
 * \{ */
const int BUILTIN_BINDING_LOCATION_OFFSET = 1000;

static constexpr int to_binding_location(const GPUUniformBuiltin builtin_uniform)
{
  return BUILTIN_BINDING_LOCATION_OFFSET + builtin_uniform;
}

static GPUUniformBuiltin to_builtin_uniform(int location)
{
  return static_cast<GPUUniformBuiltin>(location - BUILTIN_BINDING_LOCATION_OFFSET);
}

static bool is_valid_location(int location)
{
  int builtin_uniform = location - BUILTIN_BINDING_LOCATION_OFFSET;
  if (builtin_uniform < 0 || builtin_uniform >= GPU_NUM_UNIFORMS) {
    return false;
  }
  return true;
}

static constexpr UniformBuiltinStructType::AttributeBinding determine_binding_struct_1(
    const GPUUniformBuiltin builtin_uniform)
{
  UniformBuiltinStructType::AttributeBinding result = {-1, 0};

  switch (builtin_uniform) {
    case GPU_UNIFORM_MODEL:
      result.binding = to_binding_location(builtin_uniform);
      result.offset = offsetof(GPUShaderBlock3dColor, ModelMatrix);
      break;
    case GPU_UNIFORM_MVP:
      result.binding = to_binding_location(builtin_uniform);
      result.offset = offsetof(GPUShaderBlock3dColor, ModelViewProjectionMatrix);
      break;
    case GPU_UNIFORM_COLOR:
      result.binding = to_binding_location(builtin_uniform);
      result.offset = offsetof(GPUShaderBlock3dColor, color);
      break;
    case GPU_UNIFORM_CLIPPLANES:
      result.binding = to_binding_location(builtin_uniform);
      result.offset = offsetof(GPUShaderBlock3dColor, WorldClipPlanes);
      break;
    case GPU_UNIFORM_SRGB_TRANSFORM:
      result.binding = to_binding_location(builtin_uniform);
      result.offset = offsetof(GPUShaderBlock3dColor, SrgbTransform);
      break;

    default:
      break;
  };

  return result;
}

static constexpr UniformBuiltinStructType::AttributeBinding determine_binding(
    const GPUShaderBlockType struct_type, const GPUUniformBuiltin builtin_uniform)
{

  switch (struct_type) {
    case GPU_SHADER_BLOCK_CUSTOM:
    case GPU_NUM_SHADER_BLOCK_TYPES:
      return {};

    case GPU_SHADER_BLOCK_3D_COLOR:
      return determine_binding_struct_1(builtin_uniform);
  };
  return {};
}

static constexpr std::array<const UniformBuiltinStructType::AttributeBinding, GPU_NUM_UNIFORMS>
builtin_uniforms_for_struct_type(const GPUShaderBlockType struct_type)
{
  return {
      determine_binding(struct_type, GPU_UNIFORM_MODEL),
      determine_binding(struct_type, GPU_UNIFORM_VIEW),
      determine_binding(struct_type, GPU_UNIFORM_MODELVIEW),
      determine_binding(struct_type, GPU_UNIFORM_PROJECTION),
      determine_binding(struct_type, GPU_UNIFORM_VIEWPROJECTION),
      determine_binding(struct_type, GPU_UNIFORM_MVP),
      determine_binding(struct_type, GPU_UNIFORM_MODEL_INV),
      determine_binding(struct_type, GPU_UNIFORM_VIEW_INV),
      determine_binding(struct_type, GPU_UNIFORM_MODELVIEW_INV),
      determine_binding(struct_type, GPU_UNIFORM_PROJECTION_INV),
      determine_binding(struct_type, GPU_UNIFORM_VIEWPROJECTION_INV),
      determine_binding(struct_type, GPU_UNIFORM_NORMAL),
      determine_binding(struct_type, GPU_UNIFORM_ORCO),
      determine_binding(struct_type, GPU_UNIFORM_CLIPPLANES),
      determine_binding(struct_type, GPU_UNIFORM_COLOR),
      determine_binding(struct_type, GPU_UNIFORM_BASE_INSTANCE),
      determine_binding(struct_type, GPU_UNIFORM_RESOURCE_CHUNK),
      determine_binding(struct_type, GPU_UNIFORM_RESOURCE_ID),
      determine_binding(struct_type, GPU_UNIFORM_SRGB_TRANSFORM),
  };
}

static constexpr std::array<
    const std::array<const UniformBuiltinStructType::AttributeBinding, GPU_NUM_UNIFORMS>,
    GPU_NUM_SHADER_BLOCK_TYPES>
    ATTRIBUTE_BINDINGS = {
        builtin_uniforms_for_struct_type(GPU_SHADER_BLOCK_CUSTOM),
        builtin_uniforms_for_struct_type(GPU_SHADER_BLOCK_3D_COLOR),
};

static constexpr size_t data_size_for(const GPUShaderBlockType struct_type)
{

  switch (struct_type) {
    case GPU_SHADER_BLOCK_CUSTOM:
    case GPU_NUM_SHADER_BLOCK_TYPES:
      return 0;

    case GPU_SHADER_BLOCK_3D_COLOR:
      return sizeof(GPUShaderBlock3dColor);
  };
  return 0;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Struct type
 * \{ */

constexpr UniformBuiltinStructType::UniformBuiltinStructType(const GPUShaderBlockType type)
    : type(type), m_attribute_bindings(ATTRIBUTE_BINDINGS[type]), m_data_size(data_size_for(type))
{
}

bool UniformBuiltinStructType::AttributeBinding::has_binding() const
{
  return binding != -1;
}

bool UniformBuiltinStructType::has_all_builtin_uniforms(const ShaderInterface &interface) const
{
  for (int i = 0; i < GPU_NUM_UNIFORMS; i++) {
    const GPUUniformBuiltin builtin_uniform = static_cast<const GPUUniformBuiltin>(i);
    const AttributeBinding &binding = attribute_binding(builtin_uniform);
    const bool builtin_is_used = interface.builtins_[i] != -1;
    const bool has_attribute = binding.has_binding();
    if (builtin_is_used && !has_attribute) {
      return false;
    }
  }
  return true;
}

static constexpr std::array<UniformBuiltinStructType, GPU_NUM_SHADER_BLOCK_TYPES>
    STRUCT_TYPE_INFOS = {
        UniformBuiltinStructType(GPU_SHADER_BLOCK_CUSTOM),
        UniformBuiltinStructType(GPU_SHADER_BLOCK_3D_COLOR),
};

const UniformBuiltinStructType &UniformBuiltinStructType::get(const GPUShaderBlockType type)
{
  return STRUCT_TYPE_INFOS[type];
}

std::optional<const GPUShaderBlockType> find_smallest_uniform_builtin_struct(
    const ShaderInterface &interface)
{
  if (!interface.has_builtin_uniforms()) {
    return std::nullopt;
  }

  if (UniformBuiltinStructType::get(GPU_SHADER_BLOCK_3D_COLOR)
          .has_all_builtin_uniforms(interface)) {
    return std::make_optional(GPU_SHADER_BLOCK_3D_COLOR);
  }

  return std::nullopt;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Struct type
 * \{ */

UniformBuiltinStruct::UniformBuiltinStruct(const GPUShaderBlockType type)
    : m_type_info(UniformBuiltinStructType::get(type))
{
  m_data = MEM_mallocN(m_type_info.data_size(), __func__);
}

UniformBuiltinStruct::~UniformBuiltinStruct()
{
  MEM_freeN(m_data);
  m_data = nullptr;
}

bool UniformBuiltinStruct::uniform_int(int location, int comp_len, int array_size, const int *data)
{
  if (!is_valid_location(location)) {
    return false;
  }
  const GPUUniformBuiltin builtin_uniform = to_builtin_uniform(location);
  const UniformBuiltinStructType::AttributeBinding &attribute = m_type_info.attribute_binding(
      builtin_uniform);

  if (!attribute.has_binding()) {
    return false;
  }
  const size_t attribute_len = comp_len * array_size * sizeof(int);
  memcpy(((uint8_t *)m_data) + attribute.offset, data, attribute_len);
  m_flags.is_dirty = true;

  return true;
}

bool UniformBuiltinStruct::uniform_float(int location,
                                         int comp_len,
                                         int array_size,
                                         const float *data)
{
  if (!is_valid_location(location)) {
    return false;
  }
  const GPUUniformBuiltin builtin_uniform = to_builtin_uniform(location);
  const UniformBuiltinStructType::AttributeBinding &attribute = m_type_info.attribute_binding(
      builtin_uniform);

  if (!attribute.has_binding()) {
    return false;
  }
  const size_t attribute_len = comp_len * array_size * sizeof(float);
  memcpy(((uint8_t *)m_data) + attribute.offset, data, attribute_len);
  m_flags.is_dirty = true;

  return true;
}

/** \} */

}  // namespace blender::gpu
