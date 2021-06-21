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

#include <cstddef>
#include <functional>
#include <iterator>
#include <limits>
#include <optional>
#include <tuple>
#include <variant>

#include "BLI_assert.h"
#include "BLI_vector.hh"

#include "testing/testing.h"

namespace blender::generational_arena {

namespace extra {
template<typename... Ts> struct overloaded : Ts... {
  using Ts::operator()...;
};
template<typename... Ts> overloaded(Ts...) -> overloaded<Ts...>;
} /* namespace extra */

class Index;
template<typename, int64_t, typename> class Arena;

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

  inline bool operator==(const Index &other) const
  {
    return (this->index == other.index) && (this->generation == other.generation);
  }

  static inline Index invalid()
  {
    return Index(std::numeric_limits<usize>::max(), std::numeric_limits<usize>::max());
  }

  std::tuple<usize, usize> get_raw() const
  {
    return std::make_tuple(this->index, this->generation);
  }

  template<typename, int64_t, typename> friend class Arena;
};

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
 public:
  class Iterator;

 private:
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

    EntryNoExist() = default;

    EntryNoExist(usize next_free)
    {
      this->next_free = next_free;
    }

    EntryNoExist(std::optional<usize> next_free)
    {
      this->next_free = next_free;
    }
  };
  struct EntryExist {
    T value;
    usize generation;

    EntryExist(T value, usize generation) : value(value), generation(generation)
    {
    }
  };

  blender::Vector<Entry> data;
  std::optional<usize> next_free_head;
  usize generation;
  usize length;

 public:
  /* default constructor */
  Arena()
  {
    this->data = Vector<Entry>();
    this->next_free_head = std::nullopt;
    this->generation = 0;
    this->length = 0;
  }
  /* other constructors */
  Arena(const usize size) : Arena()
  {
    this->reserve(size);
  }
  /* copy constructor */
  /* move constructor */

  /* destructor */

  /* copy assignment operator */
  /* move assignment operator */
  /* other operator overloads */

  /* all public static methods */
  /* all public non-static methods */
  void reserve(const usize new_cap)
  {
    /* Must only increase capacity */
    if (new_cap < this->data.size()) {
      return;
    }

    this->data.reserve(new_cap);
    /* next_free_head is set to start of extended list
     *
     * in the extended elements, next_free is set to the next element
     *
     * last element in the extended elements's next_free is the old
     * next_free_head */
    auto const old_next_free_head = this->next_free_head;
    auto const start = this->data.size();
    for (auto i = start; i < new_cap - 1; i++) {
      this->data.append(EntryNoExist(i + 1));
    }
    this->data.append(EntryNoExist(old_next_free_head));
    this->next_free_head = start;
  }

  /* TODO(ish): add optimization by moving `value`, can be done by
   * returning value if `try_insert()` fails */
  std::optional<Index> try_insert(T value)
  {
    if (this->next_free_head) {
      auto loc = *this->next_free_head;

      if (auto entry = std::get_if<EntryNoExist>(&this->data[loc])) {
        this->next_free_head = entry->next_free;
        this->data[loc] = EntryExist(value, this->generation);
        this->length += 1;
        return Index(loc, this->generation);
      }

      /* The linked list created to
       * know where to insert next is
       * corrupted.
       * `this->next_free_head` is corrupted */
      BLI_assert_unreachable();
    }
    return std::nullopt;
  }

  Index insert(T value)
  {
    if (auto index = this->try_insert(value)) {
      return *index;
    }

    /* couldn't insert the value within reserved memory space */
    const auto reserve_cap = this->data.size() == 0 ? 1 : this->data.size();
    this->reserve(reserve_cap * 2);
    if (auto index = this->try_insert(value)) {
      return *index;
    }

    /* now that more memory has been reserved, it shouldn't fail */
    BLI_assert_unreachable();
    return Index::invalid();
  }

  /* TODO(ish): add optimization by moving `f`, can be done by
   * returning value if `try_insert()` fails */
  std::optional<Index> try_insert_with(std::function<T(Index)> f)
  {
    if (this->next_free_head) {
      auto loc = *this->next_free_head;

      if (auto entry = std::get_if<EntryNoExist>(&this->data[loc])) {
        this->next_free_head = entry->next_free;
        Index index(loc, this->generation);
        T value = f(index);
        this->data[loc] = EntryExist(value, this->generation);
        this->length += 1;

        return index;
      }

      /* The linked list created to
       * know where to insert next is
       * corrupted.
       * `this->next_free_head` is corrupted */
      BLI_assert_unreachable();
    }
    return std::nullopt;
  }

  Index insert_with(std::function<T(Index)> f)
  {
    if (auto index = this->try_insert_with(f)) {
      return *index;
    }

    /* couldn't insert the value within reserved memory space */
    const auto reserve_cap = this->data.size() == 0 ? 1 : this->data.size();
    this->reserve(reserve_cap * 2);
    if (auto index = this->try_insert_with(f)) {
      return *index;
    }

    /* now that more memory has been reserved, it shouldn't fail */
    BLI_assert_unreachable();
    return Index::invalid();
  }

  std::optional<T> remove(Index index)
  {
    if (index.index >= this->data.size()) {
      return std::nullopt;
    }

    if (auto entry = std::get_if<EntryExist>(&this->data[index.index])) {
      if (index.generation != entry->generation) {
        return std::nullopt;
      }

      /* must update the next_free list, length and generation */
      this->length -= 1;
      this->generation += 1;
      auto value = std::move(entry->value);
      this->data[index.index] = EntryNoExist(this->next_free_head);
      this->next_free_head = index.index;
      return value;
    }

    return std::nullopt;
  }

  std::optional<std::reference_wrapper<const T>> get(Index index) const
  {
    /* if index exceeds size of the container, return std::nullopt */
    if (index.index >= this->data.size()) {
      return std::nullopt;
    }

    if (auto entry = std::get_if<EntryExist>(&this->data[index.index])) {
      if (index.generation != entry->generation) {
        return std::nullopt;
      }

      return std::cref(entry->value);
    }

    return std::nullopt;
  }

  std::optional<std::reference_wrapper<T>> get(Index index)
  {
    /* if index exceeds size of the container, return std::nullopt */
    if (index.index >= this->data.size()) {
      return std::nullopt;
    }

    if (auto entry = std::get_if<EntryExist>(&this->data[index.index])) {
      if (index.generation != entry->generation) {
        return std::nullopt;
      }

      return std::ref(entry->value);
    }

    return std::nullopt;
  }

  std::optional<std::reference_wrapper<const T>> get_no_gen(usize index) const
  {
    /* if index exceeds size of the container, return std::nullopt */
    if (index >= this->data.size()) {
      return std::nullopt;
    }

    if (auto entry = std::get_if<EntryExist>(&this->data[index])) {
      return std::cref(entry->value);
    }

    return std::nullopt;
  }

  std::optional<std::reference_wrapper<T>> get_no_gen(usize index)
  {
    /* if index exceeds size of the container, return std::nullopt */
    if (index >= this->data.size()) {
      return std::nullopt;
    }

    if (auto entry = std::get_if<EntryExist>(&this->data[index])) {
      return std::ref(entry->value);
    }

    return std::nullopt;
  }

  std::optional<Index> get_no_gen_index(usize index) const
  {
    /* if index exceeds size of the container, return std::nullopt */
    if (index >= this->data.size()) {
      return std::nullopt;
    }
    if (auto entry = std::get_if<EntryExist>(&this->data[index])) {
      return Index(index, entry->generation);
    }

    return std::nullopt;
  }

  isize capacity() const
  {
    return static_cast<isize>(this->data.size());
  }

  isize size() const
  {
    return static_cast<isize>(this->length);
  }

  Iterator begin()
  {
    return Iterator(this->data.begin(), this->data.begin(), this->data.end());
  }

  Iterator end()
  {
    return Iterator(this->data.end(), this->data.begin(), this->data.end());
  }

  class Iterator {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = value_type *;
    using reference = value_type &;

   private:
    Entry *ptr;   /* points to current position */
    Entry *start; /* points to first element in the
                   * Arena::data, aka Arena::data.begin() */
    Entry *end;   /* points to last+1 element in the Arena::data, aka Arena::data.end()*/

   public:
    Iterator(Entry *ptr, Entry *start, Entry *end) : ptr(ptr), start(start), end(end)
    {
    }

    reference operator*() const
    {
      if (auto val = std::get_if<EntryExist>(this->ptr)) {
        return val->value;
      }

      BLI_assert_unreachable();

      return std::get<EntryExist>(*this->ptr).value;
    }

    pointer operator->()
    {
      return this->ptr;
    }

    /* pre fix */
    Iterator &operator++()
    {
      BLI_assert(this->ptr != this->end);
      while (true) {
        this->ptr++;

        if (this->ptr == this->end) {
          break;
        }

        if (auto val = std::get_if<EntryExist>(this->ptr)) {
          break;
        }
      }
      return *this;
    }

    Iterator &operator--()
    {
      BLI_assert(this->ptr != this->start);
      while (true) {
        this->ptr--;

        if (this->ptr == this->start) {
          break;
        }

        if (auto val = std::get_if<EntryExist>(this->ptr)) {
          break;
        }
      }
      return *this;
    }

    /* post fix */
    Iterator operator++(int)
    {
      Iterator temp = *this;
      ++(*this);
      return temp;
    }

    Iterator operator--(int)
    {
      Iterator temp = *this;
      --(*this);
      return temp;
    }

    friend bool operator==(const Iterator &a, const Iterator &b)
    {
      return a.start == b.start && a.end == b.end && a.ptr == b.ptr;
    }

    friend bool operator!=(const Iterator &a, const Iterator &b)
    {
      return a.start != b.start || a.end != b.end || a.ptr != b.ptr;
    }
  };

 protected:
  /* all protected static methods */
  /* all protected non-static methods */

 private:
  /* all private static methods */
  /* all private non-static methods */

  FRIEND_TEST(generational_arena, GetNextFreeLocations);

  blender::Vector<usize> get_next_free_locations() const
  {
    auto next_free = this->next_free_head;
    blender::Vector<usize> locs;
    locs.reserve(this->capacity() - this->size());

    while (next_free) {
      locs.append(*next_free);
      if (auto entry = std::get_if<EntryNoExist>(&this->data[*next_free])) {
        next_free = entry->next_free;
      }
      else {
        BLI_assert_unreachable();
      }
    }

    return locs;
  }
};

} /* namespace blender::generational_arena */
