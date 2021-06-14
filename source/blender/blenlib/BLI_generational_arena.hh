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
 *
 * A `blender::generation_arena<T>` is a dynamically growing
 * contiguous array for values of type T. It is designed to have
 * a similar api as `blender::Vector<T>` but with generational
 * indices. There are benefits to generational arenas.
 */
/* TODO(ish): need to complete documentation */

#include <optional>
#include <tuple>
#include <variant>

#include "BLI_vector.hh"

namespace blender {
namespace generational_arena {

template<
    /**
     * Type of the values stored in this vector. It has to be movable.
     */
    typename T,
    /**
     * The number of values that can be stored in this vector, without doing a heap allocation.
     * Sometimes it makes sense to increase this value a lot. The memory in the inline buffer is
     * not initialized when it is not needed.
     *
     * When T is large, the small buffer optimization is disabled by default to avoid large
     * unexpected allocations on the stack. It can still be enabled explicitly though.
     */
    int64_t InlineBufferCapacity = default_inline_buffer_capacity(sizeof(T)),
    /**
     * The allocator used by this vector. Should rarely be changed, except when you don't want that
     * MEM_* is used internally.
     */
    typename Allocator = GuardedAllocator>
class Arena {
  /* using declarations */
  using usize = uint64_t;
  using isize = int64_t;
  using Entry = std::variant<EntryNoExist, EntryExist>;

  /* static data members */
  /* non-static data members */
  struct EntryNoExist {
    std::optional<usize> next_free;
  };
  struct EntryExist {
    T value;
    usize generation;
  };

 public:
  /* default constructor */
  /* other constructors */
  /* copy constructor */
  /* move constructor */

  /* destructor */

  /* copy assignment operator */
  /* move assignment operator */
  /* other operator overloads */

  /* all public static methods */
  /* all public non-static methods */

 protected:
  /* all protected static methods */
  /* all protected non-static methods */

 private:
  /* all private static methods */
  /* all private non-static methods */
};

class Index {
  using usize = uint64_t;

  usize index;
  usize generation;

 public:
  Index(usize index, usize generation)
  {
    this->index = index;
    this->generation = generation;
  }

  std::tuple<usize, usize> get_raw() const
  {
    return std::make_tuple(this->index, this->generation)
  }
};

} /* namespace generational_arena */

} /* namespace blender */
