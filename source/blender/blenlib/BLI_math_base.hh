/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. */

#pragma once

/** \file
 * \ingroup bli
 */

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "BLI_math_base_safe.h"
#include "BLI_math_vector.h"
#include "BLI_utildefines.h"

#ifdef WITH_GMP
#  include "BLI_math_mpq.hh"
#endif

namespace blender::math {

#ifdef WITH_GMP
#  define BLI_ENABLE_IF_FLT(T) \
    BLI_ENABLE_IF((std::is_floating_point_v<T> || std::is_same_v<T, mpq_class>))
#else
#  define BLI_ENABLE_IF_FLT(T) BLI_ENABLE_IF((std::is_floating_point_v<T>))
#endif

#define BLI_ENABLE_IF_INT(T) BLI_ENABLE_IF((std::is_integral_v<T>))

template<typename T> inline bool is_zero(const T &a)
{
  return a == T(0);
}

template<typename T> inline bool is_any_zero(const T &a)
{
  return is_zero(a);
}

template<typename T> inline T abs(const T &a)
{
  return std::abs(a);
}

template<typename T> inline T min(const T &a, const T &b)
{
  return std::min(a, b);
}

template<typename T> inline T max(const T &a, const T &b)
{
  return std::max(a, b);
}

template<typename T> inline T clamp(const T &a, const T &min, const T &max)
{
  return std::clamp(a, min, max);
}

template<typename T, BLI_ENABLE_IF_FLT(T)> inline T mod(const T &a, const T &b)
{
  return std::fmod(a, b);
}

template<typename T, BLI_ENABLE_IF_FLT(T)> inline T safe_mod(const T &a, const T &b)
{
  return (b != 0) ? std::fmod(a, b) : 0;
}

template<typename T> inline void min_max(const T &vector, T &min_vec, T &max_vec)
{
  min_vec = min(vector, min_vec);
  max_vec = max(vector, max_vec);
}

template<typename T, BLI_ENABLE_IF_FLT(T)> inline T safe_divide(const T &a, const T &b)
{
  return (b != 0) ? a / b : T(0.0f);
}

template<typename T, BLI_ENABLE_IF_FLT(T)> inline T floor(const T &a)
{
  return std::floor(a);
}

template<typename T, BLI_ENABLE_IF_FLT(T)> inline T ceil(const T &a)
{
  return std::ceil(a);
}

template<typename T, BLI_ENABLE_IF_FLT(T)> inline T fract(const T &a)
{
  return a - std::floor(a);
}

template<typename T, typename FactorT, BLI_ENABLE_IF_FLT(T), BLI_ENABLE_IF_FLT(FactorT)>
inline T interpolate(const T &a, const T &b, const FactorT &t)
{
  return a * (1 - t) + b * t;
}

template<typename T, BLI_ENABLE_IF_FLT(T)> inline T midpoint(const T &a, const T &b)
{
  return (a + b) * 0.5;
}

#undef BLI_ENABLE_IF_FLT
#undef BLI_ENABLE_IF_INT

}  // namespace blender::math
