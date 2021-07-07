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

#include "GPU_shader.h"
#include "GPU_uniform_buffer_types.h"
#include "gpu_shader_interface.hh"

namespace blender::gpu {

/* -------------------------------------------------------------------- */
/** \name Struct type
 * \{ */

UniformBuiltinStructType::UniformBuiltinStructType(const GPUUniformBuiltinStructType type)
    : type(type)
{
}

bool UniformBuiltinStructType::has_attribute(const GPUUniformBuiltin builtin_uniform) const
{
  switch (type) {
    case GPU_UNIFORM_STRUCT_NONE:
      return false;
      break;

    case GPU_UNIFORM_STRUCT_1:
      return ELEM(builtin_uniform,
                  GPU_UNIFORM_MODEL,
                  GPU_UNIFORM_MVP,
                  GPU_UNIFORM_COLOR,
                  GPU_UNIFORM_CLIPPLANES,
                  GPU_UNIFORM_SRGB_TRANSFORM);
      break;

    case GPU_NUM_UNIFORM_STRUCTS:
      return false;
      break;
  }
  return false;
}

bool UniformBuiltinStructType::has_all_builtin_uniforms(const ShaderInterface &interface) const
{
  for (int i = 0; i < GPU_NUM_UNIFORMS; i++) {
    const GPUUniformBuiltin builtin_uniform = static_cast<const GPUUniformBuiltin>(i);
    const bool builtin_is_used = interface.builtins_[i] != -1;
    if (builtin_is_used && !has_attribute(builtin_uniform)) {
      return false;
    }
  }
  return true;
}

std::optional<const GPUUniformBuiltinStructType> find_smallest_uniform_builtin_struct(
    const ShaderInterface &interface)
{
  if (!interface.has_builtin_uniforms()) {
    return std::nullopt;
  }

  const UniformBuiltinStructType struct1(GPU_UNIFORM_STRUCT_1);

  if (struct1.has_all_builtin_uniforms(interface)) {
    return std::make_optional(struct1.type);
  }

  return std::nullopt;
}

/** \} */

}  // namespace blender::gpu
