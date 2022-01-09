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

/** \file
 * \ingroup bli
 */

#ifdef WITH_GMP

#  include <iostream>

#  include "BLI_math.h"
#  include "BLI_math_mpq.hh"
#  include "BLI_math_vector.hh"
#  include "BLI_span.hh"

namespace blender {

using mpq3 = vec3_base<mpq_class>;

uint64_t hash_mpq_class(const mpq_class &value);

namespace math {

mpq3 cross_poly(Span<mpq3> poly);

template<> inline uint64_t vector_hash(const mpq3 &vec)
{
  uint64_t hashx = hash_mpq_class(vec.x);
  uint64_t hashy = hash_mpq_class(vec.y);
  uint64_t hashz = hash_mpq_class(vec.z);
  return hashx ^ (hashy * 33) ^ (hashz * 33 * 37);
}

/**
 * Cannot do this exactly in rational arithmetic!
 * Approximate by going in and out of doubles.
 */
template<> inline mpq_class length(const mpq3 &a)
{
  return mpq_class(sqrt(length_squared(a).get_d()));
}

/**
 * The buffer avoids allocating a temporary variable.
 */
inline mpq_class distance_squared_with_buffer(const mpq3 &a, const mpq3 &b, mpq3 &buffer)
{
  buffer = a;
  buffer -= b;
  return dot(buffer, buffer);
}

/**
 * The buffer avoids allocating a temporary variable.
 */
inline mpq_class dot_with_buffer(const mpq3 &a, const mpq3 &b, mpq3 &buffer)
{
  buffer = a;
  buffer *= b;
  buffer.x += buffer.y;
  buffer.x += buffer.z;
  return buffer.x;
}

}  // namespace math

}  // namespace blender

#endif /* WITH_GMP */
