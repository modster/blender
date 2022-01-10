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

#include <array>
#include <cmath>
#include <iostream>
#include <type_traits>

#include "BLI_math_base_safe.h"
#include "BLI_math_vector.h"
#include "BLI_utildefines.h"

#ifdef WITH_GMP
#  include "BLI_math_mpq.hh"
#endif

#define ASSERT_UNIT_VECTOR(v) \
  { \
    const float _test_unit = length_squared(v); \
    BLI_assert(!(std::abs(_test_unit - 1.0f) >= BLI_ASSERT_UNIT_EPSILON) || \
               !(std::abs(_test_unit) >= BLI_ASSERT_UNIT_EPSILON)); \
  } \
  (void)0

namespace blender::math {

#define bT typename T::base_type

#ifdef WITH_GMP
#  define IS_FLOATING_POINT \
    typename std::enable_if_t< \
        std::disjunction_v<std::is_floating_point<bT>, std::is_same<bT, mpq_class>>> * = nullptr
#else
#  define IS_FLOATING_POINT \
    typename std::enable_if_t<std::is_floating_point<bT>::value> * = nullptr
#endif

#define IS_INTEGRAL typename std::enable_if_t<std::is_integral<bT>::value> * = nullptr

template<typename T> inline T abs(const T &a)
{
  T result;
  for (int i = 0; i < T::type_length; i++) {
    result[i] = a[i] >= 0 ? a[i] : -a[i];
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
  bT len;
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

template<typename T, IS_FLOATING_POINT> inline int dominant_axis(const T &a)
{
  T b = abs(a);
  return ((b.x > b.y) ? ((b.x > b.z) ? 0 : 2) : ((b.y > b.z) ? 1 : 2));
}

/** Intersections. */

template<typename T, IS_FLOATING_POINT> struct isect_result {
  enum {
    LINE_LINE_COLINEAR = -1,
    LINE_LINE_NONE = 0,
    LINE_LINE_EXACT = 1,
    LINE_LINE_CROSS = 2,
  } kind;
  bT lambda;
};

/* TODO(fclem) Should be moved to math namespace once mpq2 is using the template. */
template<typename T, IS_FLOATING_POINT>
isect_result<T> isect_seg_seg(const T &v1, const T &v2, const T &v3, const T &v4);

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

template<typename T, int Size> struct vec_struct_base {
  std::array<T, Size> values;
};

template<typename T> struct vec_struct_base<T, 2> {
  T x, y;
};

template<typename T> struct vec_struct_base<T, 3> {
  T x, y, z;
};

template<typename T> struct vec_struct_base<T, 4> {
  T x, y, z, w;
};

template<typename T, int Size> struct vec_base : public vec_struct_base<T, Size> {

  static constexpr int type_length = Size;

  typedef T base_type;
  typedef vec_base<as_uint_type<T>, Size> uint_type;

  vec_base() = default;

  explicit vec_base(uint value)
  {
    for (int i = 0; i < Size; i++) {
      (*this)[i] = static_cast<T>(value);
    }
  }

  explicit vec_base(int value)
  {
    for (int i = 0; i < Size; i++) {
      (*this)[i] = static_cast<T>(value);
    }
  }

  explicit vec_base(float value)
  {
    for (int i = 0; i < Size; i++) {
      (*this)[i] = static_cast<T>(value);
    }
  }

  explicit vec_base(double value)
  {
    for (int i = 0; i < Size; i++) {
      (*this)[i] = static_cast<T>(value);
    }
  }

#define VECTOR_ENABLE_IF_SIZE_IS(_test) \
  int S = Size, typename std::enable_if_t<S _test> * = nullptr

  template<VECTOR_ENABLE_IF_SIZE_IS(== 2)> vec_base(T _x, T _y)
  {
    (*this)[0] = _x;
    (*this)[1] = _y;
  }

  template<VECTOR_ENABLE_IF_SIZE_IS(== 3)> vec_base(T _x, T _y, T _z)
  {
    (*this)[0] = _x;
    (*this)[1] = _y;
    (*this)[2] = _z;
  }

  template<VECTOR_ENABLE_IF_SIZE_IS(== 4)> vec_base(T _x, T _y, T _z, T _w)
  {
    (*this)[0] = _x;
    (*this)[1] = _y;
    (*this)[2] = _z;
    (*this)[3] = _w;
  }

  /** Mixed scalar-vector constructors. */

  template<typename U, VECTOR_ENABLE_IF_SIZE_IS(== 3)>
  constexpr vec_base(const vec_base<U, 2> &xy, T z)
      : vec_base(static_cast<T>(xy.x), static_cast<T>(xy.y), z)
  {
  }

  template<typename U, VECTOR_ENABLE_IF_SIZE_IS(== 3)>
  constexpr vec_base(T x, const vec_base<U, 2> &yz)
      : vec_base(x, static_cast<T>(yz.x), static_cast<T>(yz.y))
  {
  }

  template<typename U, VECTOR_ENABLE_IF_SIZE_IS(== 4)>
  vec_base(vec_base<U, 3> xyz, T w)
      : vec_base(
            static_cast<T>(xyz.x), static_cast<T>(xyz.y), static_cast<T>(xyz.z), static_cast<T>(w))
  {
  }

  template<typename U, VECTOR_ENABLE_IF_SIZE_IS(== 4)>
  vec_base(T x, vec_base<U, 3> yzw)
      : vec_base(
            static_cast<T>(x), static_cast<T>(yzw.x), static_cast<T>(yzw.y), static_cast<T>(yzw.z))
  {
  }

  template<typename U, VECTOR_ENABLE_IF_SIZE_IS(== 4)>
  vec_base(vec_base<U, 2> xy, vec_base<U, 2> zw)
      : vec_base(
            static_cast<T>(xy.x), static_cast<T>(xy.y), static_cast<T>(zw.x), static_cast<T>(zw.y))
  {
  }

  template<typename U, VECTOR_ENABLE_IF_SIZE_IS(== 4)>
  vec_base(vec_base<U, 2> xy, T z, T w)
      : vec_base(static_cast<T>(xy.x), static_cast<T>(xy.y), static_cast<T>(z), static_cast<T>(w))
  {
  }

  template<typename U, VECTOR_ENABLE_IF_SIZE_IS(== 4)>
  vec_base(T x, vec_base<U, 2> yz, T w)
      : vec_base(static_cast<T>(x), static_cast<T>(yz.x), static_cast<T>(yz.y), static_cast<T>(w))
  {
  }

  template<typename U, VECTOR_ENABLE_IF_SIZE_IS(== 4)>
  vec_base(T x, T y, vec_base<U, 2> zw)
      : vec_base(static_cast<T>(x), static_cast<T>(y), static_cast<T>(zw.x), static_cast<T>(zw.y))
  {
  }

  /** Masking. */

  template<VECTOR_ENABLE_IF_SIZE_IS(>= 3)> explicit operator vec_base<T, 2>() const
  {
    return vec_base<T, 2>(UNPACK2(*this));
  }

  template<VECTOR_ENABLE_IF_SIZE_IS(>= 4)> explicit operator vec_base<T, 3>() const
  {
    return vec_base<T, 3>(UNPACK3(*this));
  }

#undef VECTOR_ENABLE_IF_SIZE_IS

  /** Conversion from pointers (from C-style vectors). */

  vec_base(const T *ptr)
  {
    for (int i = 0; i < Size; i++) {
      (*this)[i] = ptr[i];
    }
  }

  vec_base(const T (*ptr)[Size]) : vec_base(static_cast<const T *>(ptr[0]))
  {
  }

  /** Conversion from other vector types. */

  template<typename U> explicit vec_base(const vec_base<U, Size> &vec)
  {
    for (int i = 0; i < Size; i++) {
      (*this)[i] = static_cast<T>(vec[i]);
    }
  }

  /** C-style pointer dereference. */

  operator const T *() const
  {
    return reinterpret_cast<const T *>(this);
  }

  operator T *()
  {
    return reinterpret_cast<T *>(this);
  }

  /** Array access. */

  const T &operator[](int index) const
  {
    BLI_assert(index >= 0);
    BLI_assert(index < Size);
    return reinterpret_cast<const T *>(this)[index];
  }

  T &operator[](int index)
  {
    BLI_assert(index >= 0);
    BLI_assert(index < Size);
    return reinterpret_cast<T *>(this)[index];
  }

  /** Arithmetic operators. */

#define VECTOR_OP(_op) \
  vec_base result; \
  for (int i = 0; i < Size; i++) { \
    _op; \
  } \
  return result;

#define VECTOR_SELF_OP(_op) \
  for (int i = 0; i < Size; i++) { \
    _op; \
  } \
  return *this;

  friend vec_base operator+(const vec_base &a, const vec_base &b)
  {
    VECTOR_OP(result[i] = a[i] + b[i]);
  }

  friend vec_base operator+(const vec_base &a, const T &b)
  {
    VECTOR_OP(result[i] = a[i] + b);
  }

  friend vec_base operator+(const T &a, const vec_base &b)
  {
    return b + a;
  }

  vec_base &operator+=(const vec_base &b)
  {
    VECTOR_SELF_OP((*this)[i] += b[i]);
  }

  vec_base &operator+=(const T &b)
  {
    VECTOR_SELF_OP((*this)[i] += b);
  }

  friend vec_base operator-(const vec_base &a)
  {
    VECTOR_OP(result[i] = -a[i]);
  }

  friend vec_base operator-(const vec_base &a, const vec_base &b)
  {
    VECTOR_OP(result[i] = a[i] - b[i]);
  }

  friend vec_base operator-(const vec_base &a, const T &b)
  {
    VECTOR_OP(result[i] = a[i] - b);
  }

  friend vec_base operator-(const T &a, const vec_base &b)
  {
    VECTOR_OP(result[i] = a - b[i]);
  }

  vec_base &operator-=(const vec_base &b)
  {
    VECTOR_SELF_OP((*this)[i] -= b[i]);
  }

  vec_base &operator-=(const T &b)
  {
    VECTOR_SELF_OP((*this)[i] -= b);
  }

  friend vec_base operator*(const vec_base &a, const vec_base &b)
  {
    VECTOR_OP(result[i] = a[i] * b[i]);
  }

  friend vec_base operator*(const vec_base &a, T b)
  {
    VECTOR_OP(result[i] = a[i] * b);
  }

  friend vec_base operator*(T a, const vec_base &b)
  {
    return b * a;
  }

  vec_base &operator*=(T b)
  {
    VECTOR_SELF_OP((*this)[i] *= b);
  }

  vec_base &operator*=(const vec_base &b)
  {
    VECTOR_SELF_OP((*this)[i] *= b[i]);
  }

  friend vec_base operator/(const vec_base &a, const vec_base &b)
  {
    BLI_assert(!b.is_any_zero());
    VECTOR_OP(result[i] = a[i] / b[i]);
  }

  friend vec_base operator/(const vec_base &a, T b)
  {
    BLI_assert(b != T(0));
    VECTOR_OP(result[i] = a[i] / b);
  }

  friend vec_base operator/(T a, const vec_base &b)
  {
    BLI_assert(!b.is_any_zero());
    VECTOR_OP(result[i] = a / b[i]);
  }

  vec_base &operator/=(T b)
  {
    BLI_assert(b != T(0));
    VECTOR_SELF_OP((*this)[i] /= b);
  }

  vec_base &operator/=(const vec_base &b)
  {
    BLI_assert(!b.is_any_zero());
    VECTOR_SELF_OP((*this)[i] /= b[i]);
  }

  /** Binary operators. */

  INTEGRAL_OP friend vec_base operator&(const vec_base &a, const vec_base &b)
  {
    VECTOR_OP(result[i] = a[i] & b[i]);
  }

  INTEGRAL_OP friend vec_base operator&(const vec_base &a, T b)
  {
    VECTOR_OP(result[i] = a[i] & b);
  }

  INTEGRAL_OP friend vec_base operator&(T a, const vec_base &b)
  {
    return b & a;
  }

  INTEGRAL_OP vec_base &operator&=(T b)
  {
    VECTOR_SELF_OP((*this)[i] &= b);
  }

  INTEGRAL_OP vec_base &operator&=(const vec_base &b)
  {
    VECTOR_SELF_OP((*this)[i] &= b[i]);
  }

  INTEGRAL_OP friend vec_base operator|(const vec_base &a, const vec_base &b)
  {
    VECTOR_OP(result[i] = a[i] | b[i]);
  }

  INTEGRAL_OP friend vec_base operator|(const vec_base &a, T b)
  {
    VECTOR_OP(result[i] = a[i] | b);
  }

  INTEGRAL_OP friend vec_base operator|(T a, const vec_base &b)
  {
    return b | a;
  }

  INTEGRAL_OP vec_base &operator|=(T b)
  {
    VECTOR_SELF_OP((*this)[i] |= b);
  }

  INTEGRAL_OP vec_base &operator|=(const vec_base &b)
  {
    VECTOR_SELF_OP((*this)[i] |= b[i]);
  }

  INTEGRAL_OP friend vec_base operator^(const vec_base &a, const vec_base &b)
  {
    VECTOR_OP(result[i] = a[i] ^ b[i]);
  }

  INTEGRAL_OP friend vec_base operator^(const vec_base &a, T b)
  {
    VECTOR_OP(result[i] = a[i] ^ b);
  }

  INTEGRAL_OP friend vec_base operator^(T a, const vec_base &b)
  {
    return b ^ a;
  }

  INTEGRAL_OP vec_base &operator^=(T b)
  {
    VECTOR_SELF_OP((*this)[i] ^= b);
  }

  INTEGRAL_OP vec_base &operator^=(const vec_base &b)
  {
    VECTOR_SELF_OP((*this)[i] ^= b[i]);
  }

  INTEGRAL_OP friend vec_base operator~(const vec_base &a)
  {
    VECTOR_OP(result[i] = ~a[i]);
  }

  /** Modulo operators. */

  INTEGRAL_OP friend vec_base operator%(const vec_base &a, const vec_base &b)
  {
    VECTOR_OP(result[i] = a[i] % b[i]);
  }

  INTEGRAL_OP friend vec_base operator%(const vec_base &a, T b)
  {
    VECTOR_OP(result[i] = a[i] % b);
  }

  INTEGRAL_OP friend vec_base operator%(T a, const vec_base &b)
  {
    VECTOR_OP(result[i] = a % b[i]);
  }

#undef VECTOR_OP
#undef VECTOR_SELF_OP

  /** Compare. */

  friend bool operator==(const vec_base &a, const vec_base &b)
  {
    bool result = true;
    for (int i = 0; i < Size; i++) {
      result = result && (a[i] == b[i]);
    }
    return result;
  }

  friend bool operator!=(const vec_base &a, const vec_base &b)
  {
    return !(a == b);
  }

  bool is_zero() const
  {
    bool result = true;
    for (int i = 0; i < Size; i++) {
      result = result && ((*this)[i] == T(0));
    }
    return result;
  }

  bool is_any_zero() const
  {
    bool result = false;
    for (int i = 0; i < Size && result; i++) {
      result = result || ((*this)[i] == T(0));
    }
    return result;
  }

  /** Misc. */

  uint64_t hash() const
  {
    return math::vector_hash(*this);
  }

  friend std::ostream &operator<<(std::ostream &stream, const vec_base &v)
  {
    stream << "(";
    for (int i = 0; i < Size; i++) {
      stream << v[i];
      if (i != Size - 1) {
        stream << ", ";
      }
    }
    stream << ")";
    return stream;
  }
};

#undef INTEGRAL_OP

}  // namespace blender
