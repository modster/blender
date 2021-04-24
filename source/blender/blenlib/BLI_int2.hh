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
#include "BLI_int3.hh"

namespace blender {

struct int2 {
  int32_t x, y;

  int2() = default;

  int2(const int32_t *ptr) : x{ptr[0]}, y{ptr[1]}
  {
  }

  explicit int2(int32_t value) : x(value), y(value)
  {
  }

  int2(int32_t x, int32_t y) : x(x), y(y)
  {
  }

  explicit int2(const float2 &other) : x(other.x), y(other.y)
  {
  }

  int2(const int3 &other) : x(other.x), y(other.y)
  {
  }

  operator int32_t *()
  {
    return &x;
  }

  operator const int32_t *() const
  {
    return &x;
  }

  bool is_zero() const
  {
    return this->x == 0 && this->y == 0;
  }

  int2 &operator+=(const int2 &other)
  {
    x += other.x;
    y += other.y;
    return *this;
  }

  int2 &operator-=(const int2 &other)
  {
    x -= other.x;
    y -= other.y;
    return *this;
  }

  int2 &operator*=(int32_t factor)
  {
    x *= factor;
    y *= factor;
    return *this;
  }

  int2 &operator/=(int32_t divisor)
  {
    x /= divisor;
    y /= divisor;
    return *this;
  }

  friend int2 operator+(const int2 &a, const int2 &b)
  {
    return {a.x + b.x, a.y + b.y};
  }

  friend int2 operator-(const int2 &a, const int2 &b)
  {
    return {a.x - b.x, a.y - b.y};
  }

  friend int2 operator*(const int2 &a, int32_t b)
  {
    return {a.x * b, a.y * b};
  }

  friend int2 operator/(const int2 &a, int32_t b)
  {
    BLI_assert(b != 0);
    return {a.x / b, a.y / b};
  }

  friend int2 operator*(int32_t a, const int2 &b)
  {
    return b * a;
  }

  friend std::ostream &operator<<(std::ostream &stream, const int2 &v)
  {
    stream << "(" << v.x << ", " << v.y << ")";
    return stream;
  }

  static int2 clamp(const int2 &a, const int2 &min, const int2 &max)
  {
    return int2(clamp_i(a.x, min.x, max.x), clamp_i(a.y, min.y, max.y));
  }

  static int2 clamp(const int2 &a, const int32_t &min, const int32_t &max)
  {
    return int2(clamp_i(a.x, min, max), clamp_i(a.y, min, max));
  }

  friend bool operator==(const int2 &a, const int2 &b)
  {
    return a.x == b.x && a.y == b.y;
  }

  friend bool operator!=(const int2 &a, const int2 &b)
  {
    return !(a == b);
  }
};

}  // namespace blender
