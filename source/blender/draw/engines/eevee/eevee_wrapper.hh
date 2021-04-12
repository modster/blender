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
 * \ingroup eevee
 *
 * Templated wrappers to make it easier to use GPU objects in C++.
 */

#pragma once

#include "BLI_utildefines.h"
#include "GPU_uniform_buffer.h"

namespace blender::eevee {

template<
    /** Type of the values stored in this uniform buffer. */
    typename T,
    /** The number of values that can be stored in this uniform buffer. */
    int64_t len>
class StructArrayBuffer {
 private:
  T data_[len];
  GPUUniformBuf *ubo_;

 public:
  StructArrayBuffer()
  {
    ubo_ = GPU_uniformbuf_create_ex(sizeof(data_), nullptr, "StructArrayBuffer");
  }
  ~StructArrayBuffer()
  {
    GPU_uniformbuf_free(ubo_);
  }

  void push_update(void)
  {
    GPU_uniformbuf_update(ubo_, data_);
  }

  const GPUUniformBuf *ubo_get(void) const
  {
    return ubo_;
  }

  /**
   * Get the value at the given index. This invokes undefined behavior when the index is out of
   * bounds.
   */
  const T &operator[](int64_t index) const
  {
    BLI_assert(index >= 0);
    BLI_assert(index < len);
    return data_[index];
  }

  T &operator[](int64_t index)
  {
    BLI_assert(index >= 0);
    BLI_assert(index < len);
    return data_[index];
  }

  /**
   * Iterator
   */
  const T *begin() const
  {
    return data_;
  }
  const T *end() const
  {
    return data_ + len;
  }

  T *begin()
  {
    return data_;
  }
  T *end()
  {
    return data_ + len;
  }
};

/** Simpler version where data is not an array. */
template<typename T> class StructBuffer : public T {
 private:
  GPUUniformBuf *ubo_;

 public:
  StructBuffer()
  {
    ubo_ = GPU_uniformbuf_create_ex(sizeof(T), nullptr, "StructBuffer");
  }
  ~StructBuffer()
  {
    DRW_UBO_FREE_SAFE(ubo_);
  }

  void push_update(void)
  {
    T *data = static_cast<T *>(this);
    GPU_uniformbuf_update(ubo_, data);
  }

  const GPUUniformBuf *ubo_get(void) const
  {
    return ubo_;
  }

  StructBuffer<T> &operator=(const T &other)
  {
    *static_cast<T *>(this) = other;
    return *this;
  }
};

}  // namespace blender::eevee
