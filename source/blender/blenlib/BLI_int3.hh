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

#include "BLI_float3.hh"

namespace blender {

struct int3 {
  int32_t x, y, z;

  int3() = default;

  int3(const int *ptr) : x{ptr[0]}, y{ptr[1]}, z{ptr[2]}
  {
  }

  explicit int3(int value) : x(value), y(value), z(value)
  {
  }

  int3(int x, int y, int z) : x(x), y(y), z(z)
  {
  }

  int3(const int3 &other) : x(other.x), y(other.y), z(other.z)
  {
  }

  explicit int3(const float3 &other) : x(other.x), y(other.y), z(other.z)
  {
  }

  operator int *()
  {
    return &x;
  }

  operator const int *() const
  {
    return &x;
  }

  bool is_zero() const
  {
    return this->x == 0 && this->y == 0 && this->z == 0;
  }

  int3 &operator+=(const int3 &other)
  {
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
  }

  int3 &operator-=(const int3 &other)
  {
    x -= other.x;
    y -= other.y;
    z -= other.z;
    return *this;
  }

  int3 &operator*=(int factor)
  {
    x *= factor;
    y *= factor;
    z *= factor;
    return *this;
  }

  int3 &operator/=(int divisor)
  {
    x /= divisor;
    y /= divisor;
    z /= divisor;
    return *this;
  }

  friend int3 operator+(const int3 &a, const int3 &b)
  {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
  }

  friend int3 operator-(const int3 &a, const int3 &b)
  {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
  }

  friend int3 operator*(const int3 &a, int b)
  {
    return {a.x * b, a.y * b, a.z * b};
  }

  friend int3 operator/(const int3 &a, int b)
  {
    BLI_assert(b != 0);
    return {a.x / b, a.y / b, a.z / b};
  }

  friend int3 operator*(int a, const int3 &b)
  {
    return b * a;
  }

  friend std::ostream &operator<<(std::ostream &stream, const int3 &v)
  {
    stream << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return stream;
  }

  static int3 clamp(const int3 &a, const int3 &min, const int3 &max)
  {
    return int3(
        clamp_i(a.x, min.x, max.x), clamp_i(a.y, min.y, max.y), clamp_i(a.z, min.z, max.z));
  }

  static int3 clamp(const int3 &a, const int32_t &min, const int32_t &max)
  {
    return int3(clamp_i(a.x, min, max), clamp_i(a.y, min, max), clamp_i(a.z, min, max));
  }

  friend bool operator==(const int3 &a, const int3 &b)
  {
    return a.x == b.x && a.y == b.y;
  }

  friend bool operator!=(const int3 &a, const int3 &b)
  {
    return !(a == b);
  }
};

}  // namespace blender
