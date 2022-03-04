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

#include <cmath>
#include <cstdint>

#include "BLI_assert.h"
#include "BLI_math_base.h"
#include "BLI_math_matrix.h"
#include "BLI_math_vec_types.hh"

namespace blender {

/* A 2D affine transformation stored in a transformation matrix in homogeneous coordinates form.
 * The stores matrix is stored column major order. */
struct Transformation2D {
 private:
  float matrix_[3][3];

 public:
  static Transformation2D zero()
  {
    Transformation2D result;
    result.matrix_[0][0] = 0.0f;
    result.matrix_[0][1] = 0.0f;
    result.matrix_[0][2] = 0.0f;
    result.matrix_[1][0] = 0.0f;
    result.matrix_[1][1] = 0.0f;
    result.matrix_[1][2] = 0.0f;
    result.matrix_[2][0] = 0.0f;
    result.matrix_[2][1] = 0.0f;
    result.matrix_[2][2] = 0.0f;
    return result;
  }

  static Transformation2D identity()
  {
    Transformation2D result = zero();
    result.matrix_[0][0] = 1.0f;
    result.matrix_[1][1] = 1.0f;
    result.matrix_[2][2] = 1.0f;
    return result;
  }

  static Transformation2D from_translation(const float2 translation)
  {
    Transformation2D result = identity();
    result.matrix_[2][0] = translation.x;
    result.matrix_[2][1] = translation.y;
    return result;
  }

  static Transformation2D from_rotation(float rotation)
  {
    Transformation2D result = zero();
    const float cosine = std::cos(rotation);
    const float sine = std::sin(rotation);
    result.matrix_[0][0] = cosine;
    result.matrix_[0][1] = sine;
    result.matrix_[1][0] = -sine;
    result.matrix_[1][1] = cosine;
    result.matrix_[2][2] = 1.0f;
    return result;
  }

  static Transformation2D from_translation_rotation_scale(const float2 translation,
                                                          float rotation,
                                                          const float2 scale)
  {
    Transformation2D result;
    const float cosine = std::cos(rotation);
    const float sine = std::sin(rotation);
    result.matrix_[0][0] = scale.x * cosine;
    result.matrix_[0][1] = scale.y * sine;
    result.matrix_[0][2] = 0.0f;
    result.matrix_[1][0] = scale.x * -sine;
    result.matrix_[1][1] = scale.y * cosine;
    result.matrix_[1][2] = 0.0f;
    result.matrix_[2][0] = translation.x;
    result.matrix_[2][1] = translation.y;
    result.matrix_[2][2] = 1.0f;
    return result;
  }

  static Transformation2D from_normalized_axes(const float2 translation,
                                               const float2 horizontal,
                                               const float2 vertical)
  {
    BLI_ASSERT_UNIT_V2(horizontal);
    BLI_ASSERT_UNIT_V2(vertical);

    Transformation2D result;
    result.matrix_[0][0] = horizontal.x;
    result.matrix_[0][1] = horizontal.y;
    result.matrix_[0][2] = 0.0f;
    result.matrix_[1][0] = vertical.x;
    result.matrix_[1][1] = vertical.y;
    result.matrix_[1][2] = 0.0f;
    result.matrix_[2][0] = translation.x;
    result.matrix_[2][1] = translation.y;
    result.matrix_[2][2] = 1.0f;
    return result;
  }

  friend Transformation2D operator*(const Transformation2D &a, const Transformation2D &b)
  {
    Transformation2D result;
    mul_m3_m3m3(result.matrix_, a.matrix_, b.matrix_);
    return result;
  }

  void operator*=(const Transformation2D &other)
  {
    mul_m3_m3_post(matrix_, other.matrix_);
  }

  friend float2 operator*(const Transformation2D &transformation, const float2 &vector)
  {
    float2 result;
    mul_v2_m3v2(result, transformation.matrix_, vector);
    return result;
  }

  friend float2 operator*(const Transformation2D &transformation, const float (*vector)[2])
  {
    return transformation * float2(vector);
  }

  Transformation2D transposed() const
  {
    Transformation2D result;
    transpose_m3_m3(result.matrix_, matrix_);
    return result;
  }

  Transformation2D inverted() const
  {
    Transformation2D result;
    invert_m3_m3(result.matrix_, matrix_);
    return result;
  }

  Transformation2D set_pivot(float2 pivot) const
  {
    return from_translation(pivot) * *this * from_translation(-pivot);
  }

  using matrix_array = float[3][3];
  const matrix_array &matrix() const
  {
    return matrix_;
  }

  friend bool operator==(const Transformation2D &a, const Transformation2D &b)
  {
    return equals_m3m3(a.matrix_, b.matrix_);
  }
};

}  // namespace blender
