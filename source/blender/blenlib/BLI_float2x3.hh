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

struct float3x2;

/**
 * A 2x3 column major matrix.
 *
 * float2x3::values[i] is the ith column of the matrix.
 *
 * |m00 m10 m20|
 * |m01 m11 m21|
 *
 */
struct float2x3 {
  float values[3][2];

  float2x3() = default;

  float2x3(const float *matrix)
  {
    memcpy(values, matrix, sizeof(float) * 6);
  }

  float2x3(const float matrix[3][2]) : float2x3(static_cast<const float *>(matrix[0]))
  {
  }

  operator float *()
  {
    return &values[0][0];
  }

  operator const float *() const
  {
    return &values[0][0];
  }

  using c_style_float2x3 = float[3][2];
  c_style_float2x3 &ptr()
  {
    return values;
  }

  const c_style_float2x3 &ptr() const
  {
    return values;
  }

  friend float2 operator*(const float2x3 &m, const float3 &v)
  {
    float2 result;

    result[0] = m.ptr()[0][0] * v[0] + m.ptr()[1][0] * v[1] + m.ptr()[2][0] * v[2];
    result[1] = m.ptr()[0][1] * v[0] + m.ptr()[1][1] * v[1] + m.ptr()[2][1] * v[2];

    return result;
  }

  friend float2x2 operator*(const float2x3 &a, const float3x2 &b);

  friend float2x3 operator*(const float2x2 &a, const float2x3 &b)
  {
    float2x3 result;

    result.ptr()[0][0] = a.ptr()[0][0] * b.ptr()[0][0] + a.ptr()[1][0] * b.ptr()[0][1];
    result.ptr()[0][1] = a.ptr()[0][1] * b.ptr()[0][0] + a.ptr()[1][1] * b.ptr()[0][1];

    result.ptr()[1][0] = a.ptr()[0][0] * b.ptr()[1][0] + a.ptr()[1][0] * b.ptr()[1][1];
    result.ptr()[1][1] = a.ptr()[0][1] * b.ptr()[1][0] + a.ptr()[1][1] * b.ptr()[1][1];

    result.ptr()[2][0] = a.ptr()[0][0] * b.ptr()[2][0] + a.ptr()[1][0] * b.ptr()[2][1];
    result.ptr()[2][1] = a.ptr()[0][1] * b.ptr()[2][0] + a.ptr()[1][1] * b.ptr()[2][1];

    return result;
  }

  /**
   * Multiplies all the elements of `m` with `val`
   */
  friend float2x3 operator*(const float2x3 &m, const float val)
  {
    float2x3 res;

    res.ptr()[0][0] = m.ptr()[0][0] * val;
    res.ptr()[0][1] = m.ptr()[0][1] * val;

    res.ptr()[1][0] = m.ptr()[1][0] * val;
    res.ptr()[1][1] = m.ptr()[1][1] * val;

    res.ptr()[2][0] = m.ptr()[2][0] * val;
    res.ptr()[2][1] = m.ptr()[2][1] * val;

    return res;
  }

  friend float2x3 operator+(const float2x3 &m1, const float2x3 &m2)
  {
    float2x3 res;

    res.ptr()[0][0] = m1.ptr()[0][0] + m2.ptr()[0][0];
    res.ptr()[0][1] = m1.ptr()[0][1] + m2.ptr()[0][1];

    res.ptr()[1][0] = m1.ptr()[1][0] + m2.ptr()[1][0];
    res.ptr()[1][1] = m1.ptr()[1][1] + m2.ptr()[1][1];

    res.ptr()[2][0] = m1.ptr()[2][0] + m2.ptr()[2][0];
    res.ptr()[2][1] = m1.ptr()[2][1] + m2.ptr()[2][1];

    return res;
  }

  float2x3 linear_blend(const float2x3 &other, float factor) const
  {
    BLI_assert(factor >= 0.0 && factor <= 1.0);
    const float inv_factor = 1.0 - factor;
    float2x3 res;

    res.ptr()[0][0] = this->ptr()[0][0] * factor + other.ptr()[0][0] * inv_factor;
    res.ptr()[0][1] = this->ptr()[0][1] * factor + other.ptr()[0][1] * inv_factor;

    res.ptr()[1][0] = this->ptr()[1][0] * factor + other.ptr()[1][0] * inv_factor;
    res.ptr()[1][1] = this->ptr()[1][1] * factor + other.ptr()[1][1] * inv_factor;

    res.ptr()[2][0] = this->ptr()[2][0] * factor + other.ptr()[2][0] * inv_factor;
    res.ptr()[2][1] = this->ptr()[2][1] * factor + other.ptr()[2][1] * inv_factor;

    return res;
  }

  float3x2 transpose() const;

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
