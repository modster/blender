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

#include "BLI_hash.hh"

namespace blender {

template<typename T> class HeapValue {
 private:
  T *value_ = nullptr;

 public:
  HeapValue(T value)
  {
    value_ = new T(std::move(value));
  }

  HeapValue(const HeapValue &other)
  {
    if (other.value_ != nullptr) {
      value_ = new T(*other.value_);
    }
  }

  HeapValue(HeapValue &&other)
  {
    value_ = other.value_;
    other.value_ = nullptr;
  }

  ~HeapValue()
  {
    delete value_;
  }

  HeapValue &operator=(const HeapValue &other)
  {
    if (this == &other) {
      return *this;
    }
    if (value_ != nullptr) {
      if (other.value_ != nullptr) {
        *value_ = *other.value_;
      }
      else {
        delete value_;
        value_ = nullptr;
      }
    }
    else {
      if (other.value_ != nullptr) {
        value_ = new T(*other.value_);
      }
      else {
        /* Do nothing. */
      }
    }
    return *this;
  }

  HeapValue &operator=(HeapValue &&other)
  {
    if (this == &other) {
      return *this;
    }
    delete value_;
    value_ = other.value_;
    other.value_ = nullptr;
    return *this;
  }

  HeapValue &operator=(T value)
  {
    if (value_ == nullptr) {
      value_ = new T(std::move(value));
    }
    else {
      *value_ = std::move(value);
    }
  }

  operator bool() const
  {
    return value_ != nullptr;
  }

  T &operator*()
  {
    BLI_assert(value_ != nullptr);
    return *value_;
  }

  const T &operator*() const
  {
    BLI_assert(value_ != nullptr);
    return *value_;
  }

  T *operator->()
  {
    BLI_assert(value_ != nullptr);
    return value_;
  }

  const T *operator->() const
  {
    BLI_assert(value_ != nullptr);
    return value_;
  }

  T *get()
  {
    return value_;
  }

  const T *get() const
  {
    return value_;
  }

  uint64_t hash() const
  {
    if (value_ != nullptr) {
      return DefaultHash<T>{}(*value_);
    }
    return 0;
  }

  static uint64_t hash_as(const T &value)
  {
    return DefaultHash<T>{}(value);
  }

  friend bool operator==(const HeapValue &a, const HeapValue &b)
  {
    if (a.value_ == nullptr && b.value_ == nullptr) {
      return true;
    }
    if (a.value_ == nullptr) {
      return false;
    }
    if (b.value_ == nullptr) {
      return false;
    }
    return *a.value_ == *b.value_;
  }

  friend bool operator==(const HeapValue &a, const T &b)
  {
    if (a.value_ == nullptr) {
      return false;
    }
    return *a.value_ == b;
  }

  friend bool operator==(const T &a, const HeapValue &b)
  {
    return b == a;
  }

  friend bool operator!=(const HeapValue &a, const HeapValue &b)
  {
    return !(a == b);
  }

  friend bool operator!=(const HeapValue &a, const T &b)
  {
    return !(a == b);
  }

  friend bool operator!=(const T &a, const HeapValue &b)
  {
    return !(a == b);
  }
};

}  // namespace blender
