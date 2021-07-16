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

#include "BLI_float2.hh"
#include "BLI_math_matrix.h"

namespace blender {

struct float2x2 {
  float values[2][2];

  float2x2() = default;

  float2x2(const float *matrix)
  {
    memcpy(values, matrix, sizeof(float) * 4);
  }

  float2x2(const float matrix[2][2]) : float2x2(static_cast<const float *>(matrix[0]))
  {
  }

  static float2x2 identity()
  {
    float2x2 mat;
    unit_m2(mat.values);
    return mat;
  }

  operator float *()
  {
    return &values[0][0];
  }

  operator const float *() const
  {
    return &values[0][0];
  }

  using c_style_float2x2 = float[2][2];
  c_style_float2x2 &ptr()
  {
    return values;
  }

  const c_style_float2x2 &ptr() const
  {
    return values;
  }

  friend float2x2 operator*(const float2x2 &a, const float2x2 &b)
  {
    float2x2 result;
    mul_m2_m2m2(result.values, a.values, b.values);
    return result;
  }

  friend float2 operator*(const float2x2 &m, const float2 &v)
  {
    float2 result;
    mul_v2_m2v2(result, m.values, v);
    return result;
  }

  friend float2 operator*(const float2x2 &m, const float (*v)[2])
  {
    return m * float2(v);
  }

  friend float2x2 operator+(const float2x2 &m1, const float2x2 &m2)
  {
    float2x2 res;

    res.ptr()[0][0] = m1.ptr()[0][0] + m2.ptr()[0][0];
    res.ptr()[0][1] = m1.ptr()[0][1] + m2.ptr()[0][1];
    res.ptr()[1][0] = m1.ptr()[1][0] + m2.ptr()[1][0];
    res.ptr()[1][1] = m1.ptr()[1][1] + m2.ptr()[1][1];

    return float2x2(res);
  }

  uint64_t hash() const
  {
    uint64_t h = 435109;
    for (int i = 0; i < 4; i++) {
      float value = (static_cast<const float *>(values[0]))[i];
      h = h * 33 + *reinterpret_cast<const uint32_t *>(&value);
    }
    return h;
  }
};

} /* namespace blender */
