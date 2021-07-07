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
 */

/** \file
 * \ingroup gpu
 */

#pragma once

#include "BLI_sys_types.h"

#include <optional>

#include "GPU_shader.h"
#include "GPU_uniform_buffer_types.h"

#include <array>

struct GPUUniformBuf;

namespace blender {
namespace gpu {

class ShaderInterface;

#ifdef DEBUG
#  define DEBUG_NAME_LEN 64
#else
#  define DEBUG_NAME_LEN 8
#endif

/**
 * Implementation of Uniform Buffers.
 * Base class which is then specialized for each implementation (GL, VK, ...).
 */
class UniformBuf {
 protected:
  /** Data size in bytes. */
  size_t size_in_bytes_;
  /** Continuous memory block to copy to GPU. This data is owned by the UniformBuf. */
  void *data_ = NULL;
  /** Debugging name */
  char name_[DEBUG_NAME_LEN];

 public:
  UniformBuf(size_t size, const char *name);
  virtual ~UniformBuf();

  virtual void update(const void *data) = 0;
  virtual void bind(int slot) = 0;
  virtual void unbind(void) = 0;

  /** Used to defer data upload at drawing time.
   * This is useful if the thread has no context bound.
   * This transfers ownership to this UniformBuf. */
  void attach_data(void *data)
  {
    data_ = data;
  }
};

/* Syntactic sugar. */
static inline GPUUniformBuf *wrap(UniformBuf *vert)
{
  return reinterpret_cast<GPUUniformBuf *>(vert);
}
static inline UniformBuf *unwrap(GPUUniformBuf *vert)
{
  return reinterpret_cast<UniformBuf *>(vert);
}
static inline const UniformBuf *unwrap(const GPUUniformBuf *vert)
{
  return reinterpret_cast<const UniformBuf *>(vert);
}

class UniformBuiltinStructType {
 public:
  constexpr UniformBuiltinStructType(const GPUUniformBuiltinStructType type);
  static const UniformBuiltinStructType &get(const GPUUniformBuiltinStructType type);

  bool has_all_builtin_uniforms(const ShaderInterface &interface) const;

  GPUUniformBuiltinStructType type;
  struct AttributeBinding {
    int binding = -1;
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

 private:
  const std::array<const AttributeBinding, GPU_NUM_UNIFORMS> &m_attribute_bindings;
  const size_t m_data_size;
};

class UniformBuiltinStruct {
 public:
  struct Flags {
    bool is_dirty : 1;
  };

  UniformBuiltinStruct(const GPUUniformBuiltinStructType type);
  ~UniformBuiltinStruct();

  void *data() const
  {
    return m_data;
  };

  const UniformBuiltinStructType &type_info() const
  {
    return m_type_info;
  }

  bool uniform_int(int location, int comp_len, int array_size, const int *data);
  bool uniform_float(int location, int comp_len, int array_size, const float *data);

 private:
  Flags m_flags;
  const UniformBuiltinStructType &m_type_info;
  void *m_data;
};

std::optional<const GPUUniformBuiltinStructType> find_smallest_uniform_builtin_struct(
    const ShaderInterface &interface);

#undef DEBUG_NAME_LEN

}  // namespace gpu
}  // namespace blender
