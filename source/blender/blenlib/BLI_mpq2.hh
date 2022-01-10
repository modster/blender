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

#  include "BLI_math_mpq.hh"
#  include "BLI_mpq3.hh"

namespace blender {

using mpq2 = vec_base<mpq_class, 2>;

namespace math {

template<> inline uint64_t vector_hash(const mpq2 &vec)
{
  uint64_t hashx = hash_mpq_class(vec.x);
  uint64_t hashy = hash_mpq_class(vec.y);
  return hashx ^ (hashy * 33);
}

/**
 * Cannot do this exactly in rational arithmetic!
 * Approximate by going in and out of doubles.
 */
template<> inline mpq_class length(const mpq2 &a)
{
  return mpq_class(sqrt(length_squared(a).get_d()));
}

}  // namespace math

}  // namespace blender

#endif /* WITH_GMP */
