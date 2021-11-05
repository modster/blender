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
 * Copyright 2021, Blender Foundation.
 */

/** \file
 * \ingroup eevee
 *
 * Allocators that may be moved to BLI at some point.
 */

#pragma once

#include "BLI_math_bits.h"
#include "BLI_vector.hh"

namespace blender::eevee {

/**
 * Allow allocation and deletion of elements without reordering.
 * Useful to keed valid indices to items inside this allocator.
 * Type T need to implement the free_resources() method.
 */
template<typename T> class IndexedAllocator {
 private:
  Vector<T> items_;
  /** Bitmap of used items. Make search of unused slot faster. */
  Vector<uint32_t> unused_;
  /** First unused batch of items in the vector for fast reallocating. */
  int64_t unused_first_ = LONG_MAX;
  /** Unused item count in the vector for fast reallocating. */
  int64_t unused_count_ = 0;

 public:
  int64_t alloc(T &&value)
  {
    if (unused_count_ > 0) {
      /* Reclaim unused slot. */
      int64_t index = unused_first_;
      items_[index] = std::move(value);
      set_slot_used(index);

      unused_count_ -= 1;
      unused_first_ = first_unused_slot_get();
      return index;
    }
    /* Not enough place, grow the vector. */
    int64_t index = items_.append_and_get_index(value);
    int64_t size_needed = (index + 32) / 32;
    int64_t size_old = unused_.size();
    unused_.resize(size_needed);
    if (size_old < size_needed) {
      for (auto i : IndexRange(size_old, size_needed - size_old)) {
        /* Used by default. */
        unused_[i] = 0;
      }
    }
    set_slot_used(index);
    return index;
  }

  void free(int64_t index)
  {
    unused_count_ += 1;
    if (index < unused_first_) {
      unused_first_ = index;
    }
    set_slot_unused(index);
    is_slot_unused(index);
    items_[index].free_resources();
  }

  /* Pruned unused shadows at the end of the vector. */
  void resize()
  {
    while (items_.size() > 0 && is_slot_unused(items_.size() - 1)) {
      set_slot_used(items_.size() - 1);
      items_.remove_last();
      unused_count_--;
    }
    if (unused_first_ >= items_.size()) {
      /* First unused has been pruned. */
      unused_first_ = first_unused_slot_get();
    }
  }

  int64_t size() const
  {
    return items_.size() - unused_count_;
  }

  class Iterator {
   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = T;
    using pointer = const T *;
    using reference = const T &;
    using difference_type = std::ptrdiff_t;

   private:
    IndexedAllocator &allocator_;
    int64_t current_;

   public:
    constexpr explicit Iterator(IndexedAllocator &allocator, int64_t current)
        : allocator_(allocator), current_(current)
    {
    }

    constexpr Iterator &operator++()
    {
      current_++;
      while ((current_ < allocator_.items_.size()) && allocator_.is_slot_unused(current_)) {
        current_++;
      }
      return *this;
    }

    constexpr Iterator operator++(int) const
    {
      Iterator iterator = *this;
      ++*this;
      return iterator;
    }

    constexpr friend bool operator!=(const Iterator &a, const Iterator &b)
    {
      return a.current_ != b.current_;
    }

    T &operator*()
    {
      BLI_assert(allocator_.is_slot_unused(current_) == false);
      return allocator_[current_];
    }
  };

  constexpr Iterator begin()
  {
    int64_t first_used = first_used_slot_get();
    if (first_used == LONG_MAX) {
      /* Will produce no iteration. */
      first_used = items_.size();
    }
    return Iterator(*this, first_used);
  }

  constexpr Iterator end()
  {
    return Iterator(*this, items_.size());
  }

  const T &operator[](int64_t index) const
  {
    BLI_assert(is_slot_unused(index) == false);
    return items_[index];
  }

  T &operator[](int64_t index)
  {
    BLI_assert(is_slot_unused(index) == false);
    return items_[index];
  }

 private:
  int64_t first_unused_slot_get(void) const
  {
    if (unused_count_ > 0) {
      for (auto i : IndexRange(unused_.size())) {
        if (unused_[i] != 0) {
          return i * 32 + bitscan_forward_uint(unused_[i]);
        }
      }
    }
    return LONG_MAX;
  }

  int64_t first_used_slot_get(void) const
  {
    if (unused_count_ < items_.size()) {
      for (auto i : IndexRange(unused_.size())) {
        if (~unused_[i] != 0) {
          return i * 32 + bitscan_forward_uint(~unused_[i]);
        }
      }
    }
    return LONG_MAX;
  }

  bool is_slot_unused(int64_t index) const
  {
    return (unused_[index / 32] & (1u << uint32_t(index % 32))) != 0;
  }

  void set_slot_unused(int64_t index)
  {
    SET_FLAG_FROM_TEST(unused_[index / 32], true, (1u << uint32_t(index % 32)));
  }

  void set_slot_used(int64_t index)
  {
    SET_FLAG_FROM_TEST(unused_[index / 32], false, (1u << uint32_t(index % 32)));
  }
};

}  // namespace blender::eevee
