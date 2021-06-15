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
 *
 * **How it works**
 * The `Arena` has a `Vector` of `Entry`(s), an optional location of
 * next `EntryNoExist` position in the `Vector`, current generation and the
 * length (note: cannot use `Vector`'s length since any element in the
 * `Arena` can be deleted but this doesn't affect the length of the
 * vector).
 *
 * Insertion involves finding a `EntryNoExist` position, if it exists,
 * update the `Arena` with the next `EntryNoExist` position with the
 * `next_free` stored in the position that is now filled. At this
 * position, set to `EntryExist` and let the `generation` be current
 * generation value in the `Arena` and value as the value supplied by
 * the user.
 *
 * Deletion involves updating setting that location to `EntryNoExist`
 * with the `next_free` set as the `Arena`'s `next_free` and updating
 * the `Arena`'s `next_free` to the location that is to be
 * deleted. The generation should also be incremented as well as the
 * length.
 *
 * When user requests for a value using `Index`, the `generation` is
 * verified to match the generation at that `index`, if it doesn't
 * match, the value at that position was deleted and then some other
 * value was inserted which means the requested value doesn't exist at
 * that location.
 */
/* TODO(ish): need to complete documentation */

#include <optional>
#include <tuple>
#include <variant>

#include "BLI_vector.hh"

namespace blender::generational_arena {

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
  struct EntryNoExist;
  struct EntryExist;
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
    return std::make_tuple(this->index, this->generation);
  }
};

} /* namespace blender::generational_arena */
