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

#include "BLI_float2.hh"
#include "BLI_float3.hh"
#include "BLI_math_vector.h"

namespace blender {

struct float4 {
  float x, y, z, w;

  float4() = default;

  float4(const float *ptr) : x{ptr[0]}, y{ptr[1]}, z{ptr[2]}, w{ptr[3]}
  {
  }

  float4(const float (*ptr)[4]) : float4(static_cast<const float *>(ptr[0]))
  {
  }

  explicit float4(float value) : x(value), y(value), z(value), w(value)
  {
  }

  explicit float4(int value) : x(value), y(value), z(value), w(value)
  {
  }

  float4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w)
  {
  }

  float4(float3 xyz, float w) : x(xyz.x), y(xyz.y), z(xyz.z), w(w)
  {
  }

  float4(float x, float3 yzw) : x(x), y(yzw.x), z(yzw.y), w(yzw.z)
  {
  }

  float4(float2 xy, float2 zw) : x(xy.x), y(xy.y), z(zw.x), w(zw.y)
  {
  }

  float4(float2 xy, float z, float w) : x(xy.x), y(xy.y), z(z), w(w)
  {
  }

  float4(float x, float2 yz, float w) : x(x), y(yz.x), z(yz.y), w(w)
  {
  }

  float4(float x, float y, float2 zw) : x(x), y(y), z(zw.x), w(zw.y)
  {
  }

  /** Conversions. */

  explicit operator float2() const
  {
    return float2(x, y);
  }

  explicit operator float3() const
  {
    return float3(x, y, z);
  }

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
    BLI_assert(index < 4);
    return (&x)[index];
  }

  float &operator[](int64_t index)
  {
    BLI_assert(index < 4);
    return (&x)[index];
  }

  /** Arithmetic. */

  friend float4 operator+(const float4 &a, const float4 &b)
  {
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
  }

  friend float4 operator+(const float4 &a, const float &b)
  {
    return {a.x + b, a.y + b, a.z + b, a.w + b};
  }

  friend float4 operator+(const float &a, const float4 &b)
  {
    return b + a;
  }

  float4 &operator+=(const float4 &b)
  {
    x += b.x, y += b.y, z += b.z, w += b.w;
    return *this;
  }

  float4 &operator+=(const float &b)
  {
    x += b, y += b, z += b, w += b;
    return *this;
  }

  friend float4 operator-(const float4 &a)
  {
    return {-a.x, -a.y, -a.z, -a.w};
  }

  friend float4 operator-(const float4 &a, const float4 &b)
  {
    return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
  }

  friend float4 operator-(const float4 &a, const float &b)
  {
    return {a.x - b, a.y - b, a.z - b, a.w - b};
  }

  friend float4 operator-(const float &a, const float4 &b)
  {
    return {a - b.x, a - b.y, a - b.z, a - b.w};
  }

  float4 &operator-=(const float4 &b)
  {
    x -= b.x, y -= b.y, z -= b.z, w -= b.w;
    return *this;
  }

  float4 &operator-=(const float &b)
  {
    x -= b, y -= b, z -= b, w -= b;
    return *this;
  }

  friend float4 operator*(const float4 &a, const float4 &b)
  {
    return {a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};
  }

  friend float4 operator*(const float4 &a, float b)
  {
    return {a.x * b, a.y * b, a.z * b, a.w * b};
  }

  friend float4 operator*(float a, const float4 &b)
  {
    return b * a;
  }

  float4 &operator*=(float b)
  {
    x *= b, y *= b, z *= b, w *= b;
    return *this;
  }

  float4 &operator*=(const float4 &b)
  {
    x *= b.x, y *= b.y, z *= b.z, w *= b.w;
    return *this;
  }

  friend float4 operator/(const float4 &a, const float4 &b)
  {
    BLI_assert(b.x != 0.0f && b.y != 0.0f && b.z != 0.0f && b.w != 0.0f);
    return {a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w};
  }

  friend float4 operator/(const float4 &a, float b)
  {
    BLI_assert(b != 0.0f);
    return {a.x / b, a.y / b, a.z / b, a.w / b};
  }

  friend float4 operator/(float a, const float4 &b)
  {
    BLI_assert(b.x != 0.0f && b.y != 0.0f && b.z != 0.0f && b.w != 0.0f);
    return {a / b.x, a / b.y, a / b.z, a / b.w};
  }

  float4 &operator/=(float b)
  {
    BLI_assert(b != 0.0f);
    x /= b;
    y /= b;
    z /= b;
    w /= b;
    return *this;
  }

  float4 &operator/=(const float4 &b)
  {
    BLI_assert(b.x != 0.0f && b.y != 0.0f && b.z != 0.0f && b.w != 0.0f);
    x /= b.x;
    y /= b.y;
    z /= b.z;
    w /= b.w;
    return *this;
  }

  /** Compare. */

  friend bool operator==(const float4 &a, const float4 &b)
  {
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
  }

  friend bool operator!=(const float4 &a, const float4 &b)
  {
    return !(a == b);
  }

  /** Print. */

  friend std::ostream &operator<<(std::ostream &stream, const float4 &v)
  {
    stream << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")";
    return stream;
  }

  float length() const
  {
    return len_v4(*this);
  }

  static float distance(const float4 &a, const float4 &b)
  {
    return (a - b).length();
  }

  static float4 safe_divide(const float4 &a, const float b)
  {
    return (b != 0.0f) ? a / b : float4(0.0f);
  }

  static float4 interpolate(const float4 &a, const float4 &b, float t)
  {
    return a * (1 - t) + b * t;
  }

  static float4 floor(const float4 &a)
  {
    return float4(floorf(a.x), floorf(a.y), floorf(a.z), floorf(a.w));
  }

  static float4 normalize(const float4 &a)
  {
    const float t = len_v4(a);
    return (t != 0.0f) ? a / t : float4(0.0f);
  }
};

}  // namespace blender
