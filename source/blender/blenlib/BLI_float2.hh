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

#include <iostream>

#include "BLI_math_vector.h"

namespace blender {

struct float2 {
  float x, y;

  float2() = default;

  float2(const float *ptr) : x{ptr[0]}, y{ptr[1]}
  {
  }

  float2(const float (*ptr)[2]) : float2(static_cast<const float *>(ptr[0]))
  {
  }

  explicit float2(float value) : x(value), y(value)
  {
  }

  explicit float2(int value) : x(value), y(value)
  {
  }

  float2(float x, float y) : x(x), y(y)
  {
  }

  /** Conversions. */

  operator const float *() const
  {
    return &x;
  }

  operator float *()
  {
    return &x;
  }

  /** Array access. */

  const float &operator[](int64_t index) const
  {
    BLI_assert(index < 2);
    return (&x)[index];
  }

  float &operator[](int64_t index)
  {
    BLI_assert(index < 2);
    return (&x)[index];
  }

  /** Arithmetic. */

  friend float2 operator+(const float2 &a, const float2 &b)
  {
    return {a.x + b.x, a.y + b.y};
  }

  friend float2 operator+(const float2 &a, const float &b)
  {
    return {a.x + b, a.y + b};
  }

  friend float2 operator+(const float &a, const float2 &b)
  {
    return b + a;
  }

  float2 &operator+=(const float2 &b)
  {
    x += b.x;
    y += b.y;
    return *this;
  }

  float2 &operator+=(const float &b)
  {
    x += b;
    y += b;
    return *this;
  }

  friend float2 operator-(const float2 &a)
  {
    return {-a.x, -a.y};
  }

  friend float2 operator-(const float2 &a, const float2 &b)
  {
    return {a.x - b.x, a.y - b.y};
  }

  friend float2 operator-(const float2 &a, const float &b)
  {
    return {a.x - b, a.y - b};
  }

  friend float2 operator-(const float &a, const float2 &b)
  {
    return {a - b.x, a - b.y};
  }

  float2 &operator-=(const float2 &b)
  {
    x -= b.x;
    y -= b.y;
    return *this;
  }

  float2 &operator-=(const float &b)
  {
    x -= b;
    y -= b;
    return *this;
  }

  friend float2 operator*(const float2 &a, const float2 &b)
  {
    return {a.x * b.x, a.y * b.y};
  }

  friend float2 operator*(const float2 &a, float b)
  {
    return {a.x * b, a.y * b};
  }

  friend float2 operator*(float a, const float2 &b)
  {
    return b * a;
  }

  float2 &operator*=(float b)
  {
    x *= b;
    y *= b;
    return *this;
  }

  float2 &operator*=(const float2 &b)
  {
    x *= b.x;
    y *= b.y;
    return *this;
  }

  friend float2 operator/(const float2 &a, const float2 &b)
  {
    BLI_assert(b.x != 0.0f && b.y != 0.0f);
    return {a.x / b.x, a.y / b.y};
  }

  friend float2 operator/(const float2 &a, float b)
  {
    BLI_assert(b != 0.0f);
    return {a.x / b, a.y / b};
  }

  friend float2 operator/(float a, const float2 &b)
  {
    BLI_assert(b.x != 0.0f && b.y != 0.0f);
    return {a / b.x, a / b.y};
  }

  float2 &operator/=(float b)
  {
    BLI_assert(b != 0.0f);
    x /= b;
    y /= b;
    return *this;
  }

  float2 &operator/=(const float2 &b)
  {
    BLI_assert(b.x != 0.0f && b.y != 0.0f);
    x /= b.x;
    y /= b.y;
    return *this;
  }

  /** Compare. */

  friend bool operator==(const float2 &a, const float2 &b)
  {
    return a.x == b.x && a.y == b.y;
  }

  friend bool operator!=(const float2 &a, const float2 &b)
  {
    return !(a == b);
  }

  /** Print. */

  friend std::ostream &operator<<(std::ostream &stream, const float2 &v)
  {
    stream << "(" << v.x << ", " << v.y << ")";
    return stream;
  }

  uint64_t hash() const
  {
    uint64_t x1 = *reinterpret_cast<const uint32_t *>(&x);
    uint64_t x2 = *reinterpret_cast<const uint32_t *>(&y);
    return (x1 * 812519) ^ (x2 * 707951);
  }

  static float2 safe_divide(const float2 &a, const float b)
  {
    return (b != 0.0f) ? a / b : float2(0.0f);
  }

  static float2 floor(const float2 &a)
  {
    return float2(floorf(a.x), floorf(a.y));
  }

  float length() const
  {
    return len_v2(*this);
  }

  float length_squared() const
  {
    return len_squared_v2(*this);
  }

  bool is_zero() const
  {
    return this->x == 0.0f && this->y == 0.0f;
  }

  /**
   * Returns a normalized vector. The original vector is not changed.
   */
  float2 normalized() const
  {
    float2 result;
    normalize_v2_v2(result, *this);
    return result;
  }

  static float2 normalize(const float2 &vec)
  {
    float2 result;
    normalize_v2_v2(result, vec);
    return result;
  }

  static float dot(const float2 &a, const float2 &b)
  {
    return a.x * b.x + a.y * b.y;
  }

  static float2 interpolate(const float2 &a, const float2 &b, float t)
  {
    return a * (1 - t) + b * t;
  }

  static float2 abs(const float2 &a)
  {
    return float2(fabsf(a.x), fabsf(a.y));
  }

  static float distance(const float2 &a, const float2 &b)
  {
    return (a - b).length();
  }

  static float distance_squared(const float2 &a, const float2 &b)
  {
    float2 diff = a - b;
    return float2::dot(diff, diff);
  }

  struct isect_result {
    enum {
      LINE_LINE_COLINEAR = -1,
      LINE_LINE_NONE = 0,
      LINE_LINE_EXACT = 1,
      LINE_LINE_CROSS = 2,
    } kind;
    float lambda;
    float mu;
  };

  static isect_result isect_seg_seg(const float2 &v1,
                                    const float2 &v2,
                                    const float2 &v3,
                                    const float2 &v4);
};

}  // namespace blender
