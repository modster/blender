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
#include "BLI_math_matrix.h"

#include <cmath>
#include <tuple>

namespace blender {

/**
 * A 2x2 column major matrix.
 *
 * float2x2::values[i] is the ith column of the matrix.
 *
 * |m00 m10|
 * |m01 m11|
 *
 */
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

  /**
   * Multiplies all the elements of `m` with `val`
   */
  friend float2x2 operator*(const float2x2 &m, const float val)
  {
    float2x2 res;

    res.ptr()[0][0] = m.ptr()[0][0] * val;
    res.ptr()[0][1] = m.ptr()[0][1] * val;
    res.ptr()[1][0] = m.ptr()[1][0] * val;
    res.ptr()[1][1] = m.ptr()[1][1] * val;

    return res;
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

  float2x2 linear_blend(const float2x2 &other, float factor) const
  {
    BLI_assert(factor >= 0.0 && factor <= 1.0);
    const float inv_factor = 1.0 - factor;
    float2x2 res;

    res.ptr()[0][0] = this->ptr()[0][0] * factor + other.ptr()[0][0] * inv_factor;
    res.ptr()[0][1] = this->ptr()[0][1] * factor + other.ptr()[0][1] * inv_factor;
    res.ptr()[1][0] = this->ptr()[1][0] * factor + other.ptr()[1][0] * inv_factor;
    res.ptr()[1][1] = this->ptr()[1][1] * factor + other.ptr()[1][1] * inv_factor;

    return float2x2(res);
  }

  /**
   * Computes the eigen decomposition of the 2x2 matrix and returns Q
   * and Lambda
   *
   * A = Q * Lambda * Q.inverse()
   *
   * A is this matrix.
   *
   * Q is a 2x2 matrix whose ith column is the eigen vector q_i of the
   * matrix A.
   *
   * Lambda is the diagonal matrix whose diagonal elements are the
   * corresponding eigenvalues.
   *
   * Reference https://en.wikipedia.org/wiki/Eigenvalue_algorithm
   * http://www.math.harvard.edu/archive/21b_fall_04/exhibits/2dmatrices/index.html
   */
  std::tuple<float2x2, float2> eigen_decomposition() const
  {
    const auto a = this->ptr()[0][0];
    const auto b = this->ptr()[1][0];
    const auto c = this->ptr()[0][1];
    const auto d = this->ptr()[1][1];

    const auto trace = a + d;
    const auto det = a * d - b * c;

    const auto l1 = (trace + std::sqrt(trace * trace - 4.0 * det)) / 2.0;
    const auto l2 = (trace - std::sqrt(trace * trace - 4.0 * det)) / 2.0;
    const auto lambda = float2(l1, l2);

    if (c != 0.0) {
      float2x2 q;
      q.ptr()[0][0] = l1 - d;
      q.ptr()[1][0] = l2 - d;
      q.ptr()[0][1] = c;
      q.ptr()[1][1] = c;
      return {q, lambda};
    }
    if (b != 0.0) {
      float2x2 q;
      q.ptr()[0][0] = b;
      q.ptr()[1][0] = b;
      q.ptr()[0][1] = l1 - a;
      q.ptr()[1][1] = l2 - a;
      return {q, lambda};
    }

    float2x2 q;
    q.ptr()[0][0] = 1;
    q.ptr()[1][0] = 0;
    q.ptr()[0][1] = 0;
    q.ptr()[1][1] = 1;
    return {q, lambda};
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
