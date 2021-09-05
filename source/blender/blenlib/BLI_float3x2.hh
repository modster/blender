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
 */

#pragma once

#include "BLI_assert.h"
#include "BLI_float2.hh"
#include "BLI_float2x2.hh"
#include "BLI_float3.hh"

namespace blender {

/**
 * A 3x2 column major matrix.
 *
 * float3x2::values[i] is the ith column of the matrix.
 *
 * |m00 m10|
 * |m01 m11|
 * |m02 m12|
 *
 */
struct float3x2 {
  float values[2][3];

  float3x2() = default;

  float3x2(const float *matrix)
  {
    memcpy(values, matrix, sizeof(float) * 6);
  }

  float3x2(const float matrix[2][3]) : float3x2(static_cast<const float *>(matrix[0]))
  {
  }

  float3x2(const float3 &col1, const float3 &col2)
  {
    this->ptr()[0][0] = col1[0];
    this->ptr()[0][1] = col1[1];
    this->ptr()[0][2] = col1[2];

    this->ptr()[1][0] = col2[0];
    this->ptr()[1][1] = col2[1];
    this->ptr()[1][2] = col2[2];
  }

  operator float *()
  {
    return &values[0][0];
  }

  operator const float *() const
  {
    return &values[0][0];
  }

  using c_style_float3x2 = float[2][3];
  c_style_float3x2 &ptr()
  {
    return values;
  }

  const c_style_float3x2 &ptr() const
  {
    return values;
  }

  friend float3 operator*(const float3x2 &m, const float2 &v)
  {
    float3 result;

    result[0] = m.ptr()[0][0] * v[0] + m.ptr()[1][0] * v[1];
    result[1] = m.ptr()[0][1] * v[0] + m.ptr()[1][1] * v[1];
    result[2] = m.ptr()[0][2] * v[0] + m.ptr()[1][2] * v[1];

    return result;
  }

  friend float3 operator*(const float3x2 &m, const float (*v)[2])
  {
    return m * float2(v);
  }

  friend float3x2 operator*(const float3x2 &m, const float2x2 &v)
  {
    float3x2 result;

    result.ptr()[0][0] = m.ptr()[0][0] * v.ptr()[0][0] + m.ptr()[1][0] * v.ptr()[0][1];
    result.ptr()[0][1] = m.ptr()[0][1] * v.ptr()[0][0] + m.ptr()[1][1] * v.ptr()[0][1];
    result.ptr()[0][2] = m.ptr()[0][2] * v.ptr()[0][0] + m.ptr()[1][2] * v.ptr()[0][1];

    result.ptr()[1][0] = m.ptr()[0][0] * v.ptr()[1][0] + m.ptr()[1][0] * v.ptr()[1][1];
    result.ptr()[1][1] = m.ptr()[0][1] * v.ptr()[1][0] + m.ptr()[1][1] * v.ptr()[1][1];
    result.ptr()[1][2] = m.ptr()[0][2] * v.ptr()[1][0] + m.ptr()[1][2] * v.ptr()[1][1];

    return result;
  }

  /**
   * Multiplies all the elements of `m` with `val`
   */
  friend float3x2 operator*(const float3x2 &m, const float val)
  {
    float3x2 res;

    res.ptr()[0][0] = m.ptr()[0][0] * val;
    res.ptr()[0][1] = m.ptr()[0][1] * val;
    res.ptr()[0][2] = m.ptr()[0][2] * val;

    res.ptr()[1][0] = m.ptr()[1][0] * val;
    res.ptr()[1][1] = m.ptr()[1][1] * val;
    res.ptr()[1][2] = m.ptr()[1][2] * val;

    return res;
  }

  friend float3x2 operator+(const float3x2 &m1, const float3x2 &m2)
  {
    float3x2 res;

    res.ptr()[0][0] = m1.ptr()[0][0] + m2.ptr()[0][0];
    res.ptr()[0][1] = m1.ptr()[0][1] + m2.ptr()[0][1];
    res.ptr()[0][2] = m1.ptr()[0][2] + m2.ptr()[0][2];

    res.ptr()[1][0] = m1.ptr()[1][0] + m2.ptr()[1][0];
    res.ptr()[1][1] = m1.ptr()[1][1] + m2.ptr()[1][1];
    res.ptr()[1][2] = m1.ptr()[1][2] + m2.ptr()[1][2];

    return res;
  }

  float3x2 linear_blend(const float3x2 &other, float factor) const
  {
    BLI_assert(factor >= 0.0 && factor <= 1.0);
    const float inv_factor = 1.0 - factor;
    float3x2 res;

    res.ptr()[0][0] = this->ptr()[0][0] * factor + other.ptr()[0][0] * inv_factor;
    res.ptr()[0][1] = this->ptr()[0][1] * factor + other.ptr()[0][1] * inv_factor;
    res.ptr()[0][2] = this->ptr()[0][2] * factor + other.ptr()[0][2] * inv_factor;

    res.ptr()[1][0] = this->ptr()[1][0] * factor + other.ptr()[1][0] * inv_factor;
    res.ptr()[1][1] = this->ptr()[1][1] * factor + other.ptr()[1][1] * inv_factor;
    res.ptr()[1][2] = this->ptr()[1][2] * factor + other.ptr()[1][2] * inv_factor;

    return res;
  }

  uint64_t hash() const
  {
    uint64_t h = 435109;
    for (int i = 0; i < 6; i++) {
      float value = (static_cast<const float *>(values[0]))[i];
      h = h * 33 + *reinterpret_cast<const uint32_t *>(&value);
    }
    return h;
  }
};

} /* namespace blender */
