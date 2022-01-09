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
 * Copyright 2022, Blender Foundation.
 */

#pragma once

/** \file
 * \ingroup bli
 */

#include <cmath>
#include <iostream>
#include <type_traits>

#include "BLI_math_base_safe.h"
#include "BLI_math_vector.h"

#define ASSERT_UNIT_VECTOR(v) \
  { \
    const float _test_unit = length_squared(v); \
    BLI_assert(!(std::abs(_test_unit - 1.0f) >= BLI_ASSERT_UNIT_EPSILON) || \
               !(std::abs(_test_unit) >= BLI_ASSERT_UNIT_EPSILON)); \
  } \
  (void)0

namespace blender::math {

#define bT typename T::base_type
#define IS_FLOATING_POINT typename std::enable_if_t<std::is_floating_point<bT>::value> * = nullptr
#define IS_INTEGRAL typename std::enable_if_t<std::is_integral<bT>::value> * = nullptr

template<typename T> inline T abs(const T &a)
{
  T result;
  for (int i = 0; i < T::type_length; i++) {
    result[i] = std::abs(a[i]);
  }
  return result;
}

template<typename T> inline T min(const T &a, const T &b)
{
  T result;
  for (int i = 0; i < T::type_length; i++) {
    result[i] = a[i] < b[i] ? a[i] : b[i];
  }
  return result;
}

template<typename T> inline T max(const T &a, const T &b)
{
  T result;
  for (int i = 0; i < T::type_length; i++) {
    result[i] = a[i] > b[i] ? a[i] : b[i];
  }
  return result;
}

/* Always safe. */
template<typename T, IS_FLOATING_POINT> inline T mod(const T &a, const T &b)
{
  T result;
  for (int i = 0; i < T::type_length; i++) {
    result[i] = (b[i] != 0) ? std::fmod(a[i], b[i]) : 0;
  }
  return result;
}

/* Always safe. */
template<typename T, IS_FLOATING_POINT> inline T mod(const T &a, bT b)
{
  if (b == 0) {
    return T(0.0f);
  }
  T result;
  for (int i = 0; i < T::type_length; i++) {
    result[i] = std::fmod(a[i], b);
  }
  return result;
}

template<typename T> inline void min_max(const T &vector, T &min_vec, T &max_vec)
{
  min_vec = min(vector, min_vec);
  max_vec = max(vector, max_vec);
}

template<typename T, IS_FLOATING_POINT> inline T safe_divide(const T &a, const T &b)
{
  T result;
  for (int i = 0; i < T::type_length; i++) {
    result[i] = (b[i] == 0) ? 0 : a[i] / b[i];
  }
  return result;
}

template<typename T, IS_FLOATING_POINT> inline T safe_divide(const T &a, const bT b)
{
  return (b != 0) ? a / b : T(0.0f);
}

template<typename T, IS_FLOATING_POINT> inline T floor(const T &a)
{
  T result;
  for (int i = 0; i < T::type_length; i++) {
    result[i] = std::floor(a[i]);
  }
  return result;
}

template<typename T, IS_FLOATING_POINT> inline T ceil(const T &a)
{
  T result;
  for (int i = 0; i < T::type_length; i++) {
    result[i] = std::ceil(a[i]);
  }
  return result;
}

template<typename T, IS_FLOATING_POINT> inline T fract(const T &a)
{
  T result;
  for (int i = 0; i < T::type_length; i++) {
    result[i] = a[i] - std::floor(a[i]);
  }
  return result;
}

template<typename T, IS_FLOATING_POINT> inline bT dot(const T &a, const T &b)
{
  bT result = a[0] * b[0];
  for (int i = 1; i < T::type_length; i++) {
    result += a[i] * b[i];
  }
  return result;
}

template<typename T, IS_FLOATING_POINT> inline bT length_squared(const T &a)
{
  return dot(a, a);
}

template<typename T, IS_FLOATING_POINT> inline bT length(const T &a)
{
  return std::sqrt(length_squared(a));
}

template<typename T, IS_FLOATING_POINT> inline bT distance_squared(const T &a, const T &b)
{
  return length_squared(a - b);
}

template<typename T, IS_FLOATING_POINT> inline bT distance(const T &a, const T &b)
{
  return length(a - b);
}

template<typename T> uint64_t vector_hash(const T &vec)
{
  const typename T::uint_type &uvec = *reinterpret_cast<const typename T::uint_type *>(&vec);

  uint64_t result;
  result = uvec.x * uint64_t(435109);
  if constexpr (T::type_length > 1) {
    result ^= uvec.y * uint64_t(380867);
  }
  if constexpr (T::type_length > 2) {
    result ^= uvec.z * uint64_t(1059217);
  }
  if constexpr (T::type_length > 3) {
    result ^= uvec.w * uint64_t(2002613);
  }
  return result;
}

template<typename T, IS_FLOATING_POINT> inline T reflect(const T &incident, const T &normal)
{
  ASSERT_UNIT_VECTOR(normal);
  return incident - 2.0 * dot(normal, incident) * normal;
}

template<typename T, IS_FLOATING_POINT>
inline T refract(const T &incident, const T &normal, const bT eta)
{
  float dot_ni = dot(normal, incident);
  float k = 1.0f - eta * eta * (1.0f - dot_ni * dot_ni);
  if (k < 0.0f) {
    return T(0.0f);
  }
  return eta * incident - (eta * dot_ni + sqrt(k)) * normal;
}

template<typename T, IS_FLOATING_POINT> inline T project(const T &p, const T &v_proj)
{
  if (UNLIKELY(v_proj.is_zero())) {
    return T(0.0f);
  }
  return v_proj * (dot(p, v_proj) / dot(v_proj, v_proj));
}

template<typename T, IS_FLOATING_POINT>
inline T normalize_and_get_length(const T &v, bT &out_length)
{
  out_length = length_squared(v);
  /* A larger value causes normalize errors in a scaled down models with camera extreme close. */
  constexpr bT threshold = std::is_same<bT, double>::value ? 1.0e-70 : 1.0e-35f;
  if (out_length > threshold) {
    out_length = sqrt(out_length);
    return v / out_length;
  }
  /* Either the vector is small or one of it's values contained `nan`. */
  out_length = 0.0;
  return T(0.0);
}

template<typename T, IS_FLOATING_POINT> inline T normalize(const T &v)
{
  float len;
  return normalize_and_get_length(v, len);
}

template<typename T, IS_FLOATING_POINT> inline T cross(const T &a, const T &b)
{
  BLI_STATIC_ASSERT(T::type_length == 3, "");
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

template<typename T, typename std::enable_if_t<std::is_same<bT, float>::value> * = nullptr>
inline T cross_high_precision(const T &a, const T &b)
{
  BLI_STATIC_ASSERT(T::type_length == 3, "");
  return {(float)((double)a.y * b.z - (double)a.z * b.y),
          (float)((double)a.z * b.x - (double)a.x * b.z),
          (float)((double)a.x * b.y - (double)a.y * b.x)};
}

template<typename T, IS_FLOATING_POINT> inline T interpolate(const T &a, const T &b, bT t)
{
  return a * (1 - t) + b * t;
}

template<typename T, IS_FLOATING_POINT>
inline T faceforward(const T &vector, const T &incident, const T &reference)
{
  return (dot(reference, incident) < 0) ? vector : -vector;
}

#undef ASSERT_UNIT_VECTOR
#undef IS_FLOATING_POINT
#undef IS_INTEGRAL
#undef bT

}  // namespace blender::math

namespace blender {

/* clang-format off */
template<typename T>
using as_uint_type = std::conditional_t<sizeof(T) == sizeof(uint8_t), uint8_t,
                     std::conditional_t<sizeof(T) == sizeof(uint16_t), uint16_t,
                     std::conditional_t<sizeof(T) == sizeof(uint32_t), uint32_t, uint64_t>>>;
/* clang-format on */

/* FIXME(fclem): This does not works. */
#define INTEGRAL_OP /* template<typename std::enable_if_t<std::is_integral_v<bT>> * = nullptr> */

template<typename bT> struct vec2_base {

  static constexpr int type_length = 2;

  typedef bT base_type;
  typedef vec2_base<as_uint_type<bT>> uint_type;

  bT x, y;

  vec2_base() = default;

  explicit vec2_base(uint value) : x(value), y(value)
  {
  }

  explicit vec2_base(int value) : x(value), y(value)
  {
  }

  explicit vec2_base(float value) : x(value), y(value)
  {
  }

  explicit vec2_base(double value) : x(value), y(value)
  {
  }

  vec2_base(bT x, bT y) : x(x), y(y)
  {
  }

  /** Conversion from pointers (from C-style vectors). */

  vec2_base(const bT *ptr) : x(ptr[0]), y(ptr[1])
  {
  }

  vec2_base(const bT (*ptr)[2]) : vec2_base(static_cast<const bT *>(ptr[0]))
  {
  }

  /** Conversion from other vec2 types. */

  template<typename T>
  explicit vec2_base(const vec2_base<T> &vec)
      : x(static_cast<bT>(vec.x)), y(static_cast<bT>(vec.y))
  {
  }

  /** C-style pointer dereference. */

  operator const bT *() const
  {
    return &x;
  }

  operator bT *()
  {
    return &x;
  }

  /** Array access. */

  const bT &operator[](int index) const
  {
    BLI_assert(index < type_length);
    return (&x)[index];
  }

  bT &operator[](int index)
  {
    BLI_assert(index < type_length);
    return (&x)[index];
  }

  /** Arithmetic. */

  friend vec2_base operator+(const vec2_base &a, const vec2_base &b)
  {
    return {a.x + b.x, a.y + b.y};
  }

  friend vec2_base operator+(const vec2_base &a, const int &b)
  {
    return {a.x + b, a.y + b};
  }

  friend vec2_base operator+(const int &a, const vec2_base &b)
  {
    return b + a;
  }

  vec2_base &operator+=(const vec2_base &b)
  {
    x += b.x;
    y += b.y;
    return *this;
  }

  vec2_base &operator+=(const int &b)
  {
    x += b;
    y += b;
    return *this;
  }

  friend vec2_base operator-(const vec2_base &a)
  {
    return {-a.x, -a.y};
  }

  friend vec2_base operator-(const vec2_base &a, const vec2_base &b)
  {
    return {a.x - b.x, a.y - b.y};
  }

  friend vec2_base operator-(const vec2_base &a, const int &b)
  {
    return {a.x - b, a.y - b};
  }

  friend vec2_base operator-(const int &a, const vec2_base &b)
  {
    return {a - b.x, a - b.y};
  }

  vec2_base &operator-=(const vec2_base &b)
  {
    x -= b.x;
    y -= b.y;
    return *this;
  }

  vec2_base &operator-=(const int &b)
  {
    x -= b;
    y -= b;
    return *this;
  }

  friend vec2_base operator*(const vec2_base &a, const vec2_base &b)
  {
    return {a.x * b.x, a.y * b.y};
  }

  friend vec2_base operator*(const vec2_base &a, int b)
  {
    return {a.x * b, a.y * b};
  }

  friend vec2_base operator*(int a, const vec2_base &b)
  {
    return b * a;
  }

  vec2_base &operator*=(int b)
  {
    x *= b;
    y *= b;
    return *this;
  }

  vec2_base &operator*=(const vec2_base &b)
  {
    x *= b.x;
    y *= b.y;
    return *this;
  }

  friend vec2_base operator/(const vec2_base &a, const vec2_base &b)
  {
    BLI_assert(b.x != bT(0) && b.y != bT(0));
    return {a.x / b.x, a.y / b.y};
  }

  friend vec2_base operator/(const vec2_base &a, int b)
  {
    BLI_assert(b != bT(0));
    return {a.x / b, a.y / b};
  }

  friend vec2_base operator/(int a, const vec2_base &b)
  {
    BLI_assert(b.x != bT(0) && b.y != bT(0));
    return {a / b.x, a / b.y};
  }

  vec2_base &operator/=(int b)
  {
    BLI_assert(b != bT(0));
    x /= b;
    y /= b;
    return *this;
  }

  vec2_base &operator/=(const vec2_base &b)
  {
    BLI_assert(b.x != bT(0) && b.y != bT(0));
    x /= b.x;
    y /= b.y;
    return *this;
  }

  /** Binary operator. */

  INTEGRAL_OP friend vec2_base operator&(const vec2_base &a, const vec2_base &b)
  {
    return {a.x & b.x, a.y & b.y};
  }

  INTEGRAL_OP friend vec2_base operator&(const vec2_base &a, bT b)
  {
    return {a.x & b, a.y & b};
  }

  INTEGRAL_OP friend vec2_base operator&(bT a, const vec2_base &b)
  {
    return b & a;
  }

  INTEGRAL_OP vec2_base &operator&=(bT b)
  {
    x &= b;
    y &= b;
    return *this;
  }

  INTEGRAL_OP vec2_base &operator&=(const vec2_base &b)
  {
    x &= b.x;
    y &= b.y;
    return *this;
  }

  INTEGRAL_OP friend vec2_base operator|(const vec2_base &a, const vec2_base &b)
  {
    return {a.x | b.x, a.y | b.y};
  }

  INTEGRAL_OP friend vec2_base operator|(const vec2_base &a, bT b)
  {
    return {a.x | b, a.y | b};
  }

  INTEGRAL_OP friend vec2_base operator|(bT a, const vec2_base &b)
  {
    return b | a;
  }

  INTEGRAL_OP vec2_base &operator|=(bT b)
  {
    x |= b;
    y |= b;
    return *this;
  }

  INTEGRAL_OP vec2_base &operator|=(const vec2_base &b)
  {
    x |= b.x;
    y |= b.y;
    return *this;
  }

  INTEGRAL_OP friend vec2_base operator^(const vec2_base &a, const vec2_base &b)
  {
    return {a.x ^ b.x, a.y ^ b.y};
  }

  INTEGRAL_OP friend vec2_base operator^(const vec2_base &a, bT b)
  {
    return {a.x ^ b, a.y ^ b};
  }

  INTEGRAL_OP friend vec2_base operator^(bT a, const vec2_base &b)
  {
    return b ^ a;
  }

  INTEGRAL_OP vec2_base &operator^=(bT b)
  {
    x ^= b;
    y ^= b;
    return *this;
  }

  INTEGRAL_OP vec2_base &operator^=(const vec2_base &b)
  {
    x ^= b.x;
    y ^= b.y;
    return *this;
  }

  INTEGRAL_OP friend vec2_base operator~(const vec2_base &a)
  {
    return {~a.x, ~a.y};
  }

  /** Modulo operator. */

  INTEGRAL_OP friend vec2_base operator%(const vec2_base &a, const vec2_base &b)
  {
    return {a.x % b.x, a.y % b.y};
  }

  INTEGRAL_OP friend vec2_base operator%(const vec2_base &a, bT b)
  {
    return {a.x % b, a.y % b};
  }

  INTEGRAL_OP friend vec2_base operator%(bT a, const vec2_base &b)
  {
    return {a % b.x, a % b.y};
  }

  /** Compare. */

  friend bool operator==(const vec2_base &a, const vec2_base &b)
  {
    return a.x == b.x && a.y == b.y;
  }

  friend bool operator!=(const vec2_base &a, const vec2_base &b)
  {
    return !(a == b);
  }

  bool is_zero() const
  {
    return x == 0 && y == 0;
  }

  uint64_t hash() const
  {
    return math::vector_hash(*this);
  }

  /** Print. */

  friend std::ostream &operator<<(std::ostream &stream, const vec2_base &v)
  {
    stream << "(" << v.x << ", " << v.y << ")";
    return stream;
  }

  /** Intersections. */

  struct isect_result {
    enum {
      LINE_LINE_COLINEAR = -1,
      LINE_LINE_NONE = 0,
      LINE_LINE_EXACT = 1,
      LINE_LINE_CROSS = 2,
    } kind;
    bT lambda;
  };

  static isect_result isect_seg_seg(const vec2_base &v1,
                                    const vec2_base &v2,
                                    const vec2_base &v3,
                                    const vec2_base &v4);
};

template<typename bT> struct vec3_base {

  static constexpr int type_length = 3;

  typedef bT base_type;
  typedef vec3_base<as_uint_type<bT>> uint_type;

  bT x, y, z;

  vec3_base() = default;

  explicit vec3_base(uint value) : x(value), y(value), z(value)
  {
  }

  explicit vec3_base(int value) : x(value), y(value), z(value)
  {
  }

  explicit vec3_base(float value) : x(value), y(value), z(value)
  {
  }

  explicit vec3_base(double value) : x(value), y(value), z(value)
  {
  }

  vec3_base(bT x, bT y, bT z) : x(x), y(y), z(z)
  {
  }

  /** Conversion from pointers (from C-style vectors). */

  vec3_base(const bT *ptr) : x(ptr[0]), y(ptr[1]), z(ptr[2])
  {
  }

  vec3_base(const bT (*ptr)[3]) : vec3_base(static_cast<const bT *>(ptr[0]))
  {
  }

  /** Conversion from other vec3 types. */

  template<typename T>
  explicit vec3_base(const vec3_base<T> &vec)
      : x(static_cast<bT>(vec.x)), y(static_cast<bT>(vec.y)), z(static_cast<bT>(vec.z))
  {
  }

  /** Mixed scalar-vector constructors. */

  template<typename T>
  constexpr vec3_base(const vec2_base<T> &xy, bT z)
      : x(static_cast<bT>(xy.x)), y(static_cast<bT>(xy.y)), z(z)
  {
  }

  template<typename T>
  constexpr vec3_base(bT x, const vec2_base<T> &yz)
      : x(x), y(static_cast<bT>(yz.x)), z(static_cast<bT>(yz.y))
  {
  }

  /** Masking. */

  explicit operator vec2_base<bT>() const
  {
    return vec2_base<bT>(x, y);
  }

  /** C-style pointer dereference. */

  operator const bT *() const
  {
    return &x;
  }

  operator bT *()
  {
    return &x;
  }

  /** Array access. */

  const bT &operator[](int64_t index) const
  {
    BLI_assert(index < type_length);
    return (&x)[index];
  }

  bT &operator[](int64_t index)
  {
    BLI_assert(index < type_length);
    return (&x)[index];
  }

  /** Arithmetic. */

  friend vec3_base operator+(const vec3_base &a, const vec3_base &b)
  {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
  }

  friend vec3_base operator+(const vec3_base &a, const bT &b)
  {
    return {a.x + b, a.y + b, a.z + b};
  }

  friend vec3_base operator+(const bT &a, const vec3_base &b)
  {
    return b + a;
  }

  vec3_base &operator+=(const vec3_base &b)
  {
    x += b.x;
    y += b.y;
    z += b.z;
    return *this;
  }

  vec3_base &operator+=(const bT &b)
  {
    x += b;
    y += b;
    z += b;
    return *this;
  }

  friend vec3_base operator-(const vec3_base &a)
  {
    return {-a.x, -a.y, -a.z};
  }

  friend vec3_base operator-(const vec3_base &a, const vec3_base &b)
  {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
  }

  friend vec3_base operator-(const vec3_base &a, const bT &b)
  {
    return {a.x - b, a.y - b, a.z - b};
  }

  friend vec3_base operator-(const bT &a, const vec3_base &b)
  {
    return {a - b.x, a - b.y, a - b.z};
  }

  vec3_base &operator-=(const vec3_base &b)
  {
    x -= b.x;
    y -= b.y;
    z -= b.z;
    return *this;
  }

  vec3_base &operator-=(const bT &b)
  {
    x -= b;
    y -= b;
    z -= b;
    return *this;
  }

  friend vec3_base operator*(const vec3_base &a, const vec3_base &b)
  {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
  }

  friend vec3_base operator*(const vec3_base &a, bT b)
  {
    return {a.x * b, a.y * b, a.z * b};
  }

  friend vec3_base operator*(bT a, const vec3_base &b)
  {
    return b * a;
  }

  vec3_base &operator*=(bT b)
  {
    x *= b;
    y *= b;
    z *= b;
    return *this;
  }

  vec3_base &operator*=(const vec3_base &b)
  {
    x *= b.x;
    y *= b.y;
    z *= b.z;
    return *this;
  }

  friend vec3_base operator/(const vec3_base &a, const vec3_base &b)
  {
    BLI_assert(b.x != bT(0) && b.y != bT(0) && b.z != bT(0));
    return {a.x / b.x, a.y / b.y, a.z / b.z};
  }

  friend vec3_base operator/(const vec3_base &a, bT b)
  {
    BLI_assert(b != bT(0));
    return {a.x / b, a.y / b, a.z / b};
  }

  friend vec3_base operator/(bT a, const vec3_base &b)
  {
    BLI_assert(b.x != bT(0) && b.y != bT(0) && b.z != bT(0));
    return {a / b.x, a / b.y, a / b.z};
  }

  vec3_base &operator/=(bT b)
  {
    BLI_assert(b != bT(0));
    x /= b;
    y /= b;
    z /= b;
    return *this;
  }

  vec3_base &operator/=(const vec3_base &b)
  {
    BLI_assert(b.x != bT(0) && b.y != bT(0) && b.z != bT(0));
    x /= b.x;
    y /= b.y;
    z /= b.z;
    return *this;
  }

  /** Binary operator. */

  INTEGRAL_OP friend vec3_base operator&(const vec3_base &a, const vec3_base &b)
  {
    return {a.x & b.x, a.y & b.y, a.z & b.z};
  }

  INTEGRAL_OP friend vec3_base operator&(const vec3_base &a, bT b)
  {
    return {a.x & b, a.y & b, a.z & b};
  }

  INTEGRAL_OP friend vec3_base operator&(bT a, const vec3_base &b)
  {
    return b & a;
  }

  INTEGRAL_OP vec3_base &operator&=(bT b)
  {
    x &= b;
    y &= b;
    z &= b;
    return *this;
  }

  INTEGRAL_OP vec3_base &operator&=(const vec3_base &b)
  {
    x &= b.x;
    y &= b.y;
    z &= b.z;
    return *this;
  }

  INTEGRAL_OP friend vec3_base operator|(const vec3_base &a, const vec3_base &b)
  {
    return {a.x | b.x, a.y | b.y, a.z | b.z};
  }

  INTEGRAL_OP friend vec3_base operator|(const vec3_base &a, bT b)
  {
    return {a.x | b, a.y | b, a.z | b};
  }

  INTEGRAL_OP friend vec3_base operator|(bT a, const vec3_base &b)
  {
    return b | a;
  }

  INTEGRAL_OP vec3_base &operator|=(bT b)
  {
    x |= b;
    y |= b;
    z |= b;
    return *this;
  }

  INTEGRAL_OP vec3_base &operator|=(const vec3_base &b)
  {
    x |= b.x;
    y |= b.y;
    z |= b.z;
    return *this;
  }

  INTEGRAL_OP friend vec3_base operator^(const vec3_base &a, const vec3_base &b)
  {
    return {a.x ^ b.x, a.y ^ b.y, a.z ^ b.z};
  }

  INTEGRAL_OP friend vec3_base operator^(const vec3_base &a, bT b)
  {
    return {a.x ^ b, a.y ^ b, a.z ^ b};
  }

  INTEGRAL_OP friend vec3_base operator^(bT a, const vec3_base &b)
  {
    return b ^ a;
  }

  INTEGRAL_OP vec3_base &operator^=(bT b)
  {
    x ^= b;
    y ^= b;
    z ^= b;
    return *this;
  }

  INTEGRAL_OP vec3_base &operator^=(const vec3_base &b)
  {
    x ^= b.x;
    y ^= b.y;
    z ^= b.z;
    return *this;
  }

  INTEGRAL_OP friend vec3_base operator~(const vec3_base &a)
  {
    return {~a.x, ~a.y, ~a.z};
  }

  /** Modulo operator. */

  INTEGRAL_OP friend vec3_base operator%(const vec3_base &a, const vec3_base &b)
  {
    return {a.x % b.x, a.y % b.y, a.z % b.z};
  }

  INTEGRAL_OP friend vec3_base operator%(const vec3_base &a, bT b)
  {
    return {a.x % b, a.y % b, a.z % b};
  }

  INTEGRAL_OP friend vec3_base operator%(bT a, const vec3_base &b)
  {
    return {a % b.x, a % b.y, a % b.z};
  }

  /** Compare. */

  friend bool operator==(const vec3_base &a, const vec3_base &b)
  {
    return a.x == b.x && a.y == b.y && a.z == b.z;
  }

  friend bool operator!=(const vec3_base &a, const vec3_base &b)
  {
    return !(a == b);
  }

  bool is_zero() const
  {
    return x == 0 && y == 0 && z == 0;
  }

  uint64_t hash() const
  {
    return math::vector_hash(*this);
  }

  /** Print. */

  friend std::ostream &operator<<(std::ostream &stream, const vec3_base &v)
  {
    stream << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return stream;
  }
};

template<typename bT> struct vec4_base {

  static constexpr int type_length = 4;

  typedef bT base_type;
  typedef vec4_base<as_uint_type<bT>> uint_type;

  bT x, y, z, w;

  vec4_base() = default;

  explicit vec4_base(uint value) : x(value), y(value), z(value), w(value)
  {
  }

  explicit vec4_base(int value) : x(value), y(value), z(value), w(value)
  {
  }

  explicit vec4_base(float value) : x(value), y(value), z(value), w(value)
  {
  }

  explicit vec4_base(double value) : x(value), y(value), z(value), w(value)
  {
  }

  vec4_base(bT x, bT y, bT z, bT w) : x(x), y(y), z(z), w(w)
  {
  }

  /** Conversion from pointers (from C-style vectors). */

  vec4_base(const bT *ptr) : x(ptr[0]), y(ptr[1]), z(ptr[2]), w(ptr[3])
  {
  }

  vec4_base(const bT (*ptr)[4]) : vec4_base(static_cast<const bT *>(ptr[0]))
  {
  }

  /** Conversion from other vec4 types. */

  template<typename T>
  explicit vec4_base(const vec4_base<T> &vec)
      : x(static_cast<bT>(vec.x)),
        y(static_cast<bT>(vec.y)),
        z(static_cast<bT>(vec.z)),
        w(static_cast<bT>(vec.w))
  {
  }

  /** Mixed scalar-vector constructors. */

  template<typename T>
  vec4_base(vec3_base<T> xyz, bT w)
      : x(static_cast<bT>(xyz.x)),
        y(static_cast<bT>(xyz.y)),
        z(static_cast<bT>(xyz.z)),
        w(static_cast<bT>(w))
  {
  }

  template<typename T>
  vec4_base(bT x, vec3_base<T> yzw)
      : x(static_cast<bT>(x)),
        y(static_cast<bT>(yzw.x)),
        z(static_cast<bT>(yzw.y)),
        w(static_cast<bT>(yzw.z))
  {
  }

  template<typename T>
  vec4_base(vec2_base<T> xy, vec2_base<T> zw)
      : x(static_cast<bT>(xy.x)),
        y(static_cast<bT>(xy.y)),
        z(static_cast<bT>(zw.x)),
        w(static_cast<bT>(zw.y))
  {
  }

  template<typename T>
  vec4_base(vec2_base<T> xy, bT z, bT w)
      : x(static_cast<bT>(xy.x)),
        y(static_cast<bT>(xy.y)),
        z(static_cast<bT>(z)),
        w(static_cast<bT>(w))
  {
  }

  template<typename T>
  vec4_base(bT x, vec2_base<T> yz, bT w)
      : x(static_cast<bT>(x)),
        y(static_cast<bT>(yz.x)),
        z(static_cast<bT>(yz.y)),
        w(static_cast<bT>(w))
  {
  }

  template<typename T>
  vec4_base(bT x, bT y, vec2_base<T> zw)
      : x(static_cast<bT>(x)),
        y(static_cast<bT>(y)),
        z(static_cast<bT>(zw.x)),
        w(static_cast<bT>(zw.y))
  {
  }

  /** Masking. */

  explicit operator vec2_base<bT>() const
  {
    return vec2_base<bT>(x, y);
  }

  explicit operator vec3_base<bT>() const
  {
    return vec3_base<bT>(x, y, z);
  }

  /** C-style pointer dereference. */

  operator const bT *() const
  {
    return &x;
  }

  operator bT *()
  {
    return &x;
  }

  /** Array access. */

  const bT &operator[](int64_t index) const
  {
    BLI_assert(index < 4);
    return (&x)[index];
  }

  bT &operator[](int64_t index)
  {
    BLI_assert(index < 4);
    return (&x)[index];
  }

  /** Arithmetic. */

  friend vec4_base operator+(const vec4_base &a, const vec4_base &b)
  {
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
  }

  friend vec4_base operator+(const vec4_base &a, const bT &b)
  {
    return {a.x + b, a.y + b, a.z + b, a.w + b};
  }

  friend vec4_base operator+(const bT &a, const vec4_base &b)
  {
    return b + a;
  }

  vec4_base &operator+=(const vec4_base &b)
  {
    x += b.x;
    y += b.y;
    z += b.z;
    w += b.w;
    return *this;
  }

  vec4_base &operator+=(const bT &b)
  {
    x += b;
    y += b;
    z += b;
    w += b;
    return *this;
  }

  friend vec4_base operator-(const vec4_base &a)
  {
    return {-a.x, -a.y, -a.z, -a.w};
  }

  friend vec4_base operator-(const vec4_base &a, const vec4_base &b)
  {
    return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
  }

  friend vec4_base operator-(const vec4_base &a, const bT &b)
  {
    return {a.x - b, a.y - b, a.z - b, a.w - b};
  }

  friend vec4_base operator-(const bT &a, const vec4_base &b)
  {
    return {a - b.x, a - b.y, a - b.z, a - b.w};
  }

  vec4_base &operator-=(const vec4_base &b)
  {
    x -= b.x;
    y -= b.y;
    z -= b.z;
    w -= b.w;
    return *this;
  }

  vec4_base &operator-=(const bT &b)
  {
    x -= b;
    y -= b;
    z -= b;
    w -= b;
    return *this;
  }

  friend vec4_base operator*(const vec4_base &a, const vec4_base &b)
  {
    return {a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};
  }

  friend vec4_base operator*(const vec4_base &a, bT b)
  {
    return {a.x * b, a.y * b, a.z * b, a.w * b};
  }

  friend vec4_base operator*(bT a, const vec4_base &b)
  {
    return b * a;
  }

  vec4_base &operator*=(bT b)
  {
    x *= b;
    y *= b;
    z *= b;
    w *= b;
    return *this;
  }

  vec4_base &operator*=(const vec4_base &b)
  {
    x *= b.x;
    y *= b.y;
    z *= b.z;
    w *= b.w;
    return *this;
  }

  friend vec4_base operator/(const vec4_base &a, const vec4_base &b)
  {
    BLI_assert(b.x != 0.0f && b.y != 0.0f && b.z != 0.0f && b.w != 0.0f);
    return {a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w};
  }

  friend vec4_base operator/(const vec4_base &a, bT b)
  {
    BLI_assert(b != 0.0f);
    return {a.x / b, a.y / b, a.z / b, a.w / b};
  }

  friend vec4_base operator/(bT a, const vec4_base &b)
  {
    BLI_assert(b.x != 0.0f && b.y != 0.0f && b.z != 0.0f && b.w != 0.0f);
    return {a / b.x, a / b.y, a / b.z, a / b.w};
  }

  vec4_base &operator/=(bT b)
  {
    BLI_assert(b != 0.0f);
    x /= b;
    y /= b;
    z /= b;
    w /= b;
    return *this;
  }

  vec4_base &operator/=(const vec4_base &b)
  {
    BLI_assert(b.x != 0.0f && b.y != 0.0f && b.z != 0.0f && b.w != 0.0f);
    x /= b.x;
    y /= b.y;
    z /= b.z;
    w /= b.w;
    return *this;
  }

  /** Binary operator. */

  INTEGRAL_OP friend vec4_base operator&(const vec4_base &a, const vec4_base &b)
  {
    return {a.x & b.x, a.y & b.y, a.z & b.z, a.w & b.w};
  }

  INTEGRAL_OP friend vec4_base operator&(const vec4_base &a, bT b)
  {
    return {a.x & b, a.y & b, a.z & b, a.w & b};
  }

  INTEGRAL_OP friend vec4_base operator&(bT a, const vec4_base &b)
  {
    return b & a;
  }

  INTEGRAL_OP vec4_base &operator&=(bT b)
  {
    x &= b;
    y &= b;
    z &= b;
    w &= b;
    return *this;
  }

  INTEGRAL_OP vec4_base &operator&=(const vec4_base &b)
  {
    x &= b.x;
    y &= b.y;
    z &= b.z;
    w &= b.w;
    return *this;
  }

  INTEGRAL_OP friend vec4_base operator|(const vec4_base &a, const vec4_base &b)
  {
    return {a.x | b.x, a.y | b.y, a.z | b.z, a.w | b.w};
  }

  INTEGRAL_OP friend vec4_base operator|(const vec4_base &a, bT b)
  {
    return {a.x | b, a.y | b, a.z | b, a.w | b};
  }

  INTEGRAL_OP friend vec4_base operator|(bT a, const vec4_base &b)
  {
    return b | a;
  }

  INTEGRAL_OP vec4_base &operator|=(bT b)
  {
    x |= b;
    y |= b;
    z |= b;
    w |= b;
    return *this;
  }

  INTEGRAL_OP vec4_base &operator|=(const vec4_base &b)
  {
    x |= b.x;
    y |= b.y;
    z |= b.z;
    w |= b.w;
    return *this;
  }

  INTEGRAL_OP friend vec4_base operator^(const vec4_base &a, const vec4_base &b)
  {
    return {a.x ^ b.x, a.y ^ b.y, a.z ^ b.z, a.w ^ b.w};
  }

  INTEGRAL_OP friend vec4_base operator^(const vec4_base &a, bT b)
  {
    return {a.x ^ b, a.y ^ b, a.z ^ b, a.w ^ b};
  }

  INTEGRAL_OP friend vec4_base operator^(bT a, const vec4_base &b)
  {
    return b ^ a;
  }

  INTEGRAL_OP vec4_base &operator^=(bT b)
  {
    x ^= b;
    y ^= b;
    z ^= b;
    w ^= b;
    return *this;
  }

  INTEGRAL_OP vec4_base &operator^=(const vec4_base &b)
  {
    x ^= b.x;
    y ^= b.y;
    z ^= b.z;
    w ^= b.w;
    return *this;
  }

  INTEGRAL_OP friend vec4_base operator~(const vec4_base &a)
  {
    return {~a.x, ~a.y, ~a.z, ~a.w};
  }

  /** Modulo operator. */

  INTEGRAL_OP friend vec4_base operator%(const vec4_base &a, const vec4_base &b)
  {
    return {a.x % b.x, a.y % b.y, a.z % b.z, a.w % b.w};
  }

  INTEGRAL_OP friend vec4_base operator%(const vec4_base &a, bT b)
  {
    return {a.x % b, a.y % b, a.z % b, a.w % b};
  }

  INTEGRAL_OP friend vec4_base operator%(bT a, const vec4_base &b)
  {
    return {a % b.x, a % b.y, a % b.z, a % b.w};
  }

  /** Compare. */

  friend bool operator==(const vec4_base &a, const vec4_base &b)
  {
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
  }

  friend bool operator!=(const vec4_base &a, const vec4_base &b)
  {
    return !(a == b);
  }

  bool is_zero() const
  {
    return x == 0 && y == 0 && z == 0 && w == 0;
  }

  uint64_t hash() const
  {
    return math::vector_hash(*this);
  }

  /** Print. */

  friend std::ostream &operator<<(std::ostream &stream, const vec4_base &v)
  {
    stream << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")";
    return stream;
  }
};

#undef INTEGRAL_OP

}  // namespace blender
