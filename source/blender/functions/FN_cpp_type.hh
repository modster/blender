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
 * \ingroup fn
 *
 * The CPPType class is the core of the runtime-type-system used by the functions system. It can
 * represent C++ types that are default-constructible, destructible, movable, copyable,
 * equality comparable and hashable. In the future we might want to make some of these properties
 * optional.
 *
 * Every type has a size and an alignment. Every function dealing with C++ types in a generic way,
 * has to make sure that alignment rules are followed. The methods provided by a CPPType instance
 * will check for correct alignment as well.
 *
 * Every type has a name that is for debugging purposes only. It should not be used as identifier.
 *
 * To check if two instances of CPPType represent the same type, only their pointers have to be
 * compared. Any C++ type has at most one corresponding CPPType instance.
 *
 * A CPPType instance comes with many methods that allow dealing with types in a generic way. Most
 * methods come in three variants. Using the construct-default methods as example:
 *  - construct_default(void *ptr):
 *      Constructs a single instance of that type at the given pointer.
 *  - construct_default_n(void *ptr, int64_t n):
 *      Constructs n instances of that type in an array that starts at the given pointer.
 *  - construct_default_indices(void *ptr, IndexMask mask):
 *      Constructs multiple instances of that type in an array that starts at the given pointer.
 *      Only the indices referenced by `mask` will by constructed.
 *
 * In some cases default-construction does nothing (e.g. for trivial types like int). The
 * `default_value` method provides some default value anyway that can be copied instead. What the
 * default value is, depends on the type. Usually it is something like 0 or an empty string.
 *
 *
 * Implementation Considerations
 * -----------------------------
 *
 * Concepts like inheritance are currently not captured by this system. This is not because it is
 * not possible, but because it was not necessary to add this complexity yet.
 *
 * One could also implement CPPType itself using virtual inheritance. However, I found the approach
 * used now with explicit function pointers to work better. Here are some reasons:
 *  - If CPPType would be inherited once for every used C++ type, we would get a lot of classes
 *    that would only be instanced once each.
 *  - Methods like `construct_default` that operate on a single instance have to be fast. Even this
 *    one necessary indirection using function pointers adds a lot of overhead. If all methods were
 *    virtual, there would be a second level of indirection that increases the overhead even more.
 *  - If it becomes necessary, we could pass the function pointers to C functions more easily than
 *    pointers to virtual member functions.
 */

#include "BLI_hash.hh"
#include "BLI_index_mask.hh"
#include "BLI_math_base.h"
#include "BLI_string_ref.hh"
#include "BLI_utility_mixins.hh"

namespace blender::fn {

struct CPPTypeMembers {
  int64_t size = 0;
  int64_t alignment = 0;
  uintptr_t alignment_mask = 0;
  bool is_trivially_destructible = false;
  bool has_special_member_functions = false;

  void (*construct_default)(void *ptr) = nullptr;
  void (*construct_default_indices)(void *ptr, IndexMask mask) = nullptr;

  void (*destruct)(void *ptr) = nullptr;
  void (*destruct_indices)(void *ptr, IndexMask mask) = nullptr;

  void (*copy_to_initialized)(const void *src, void *dst) = nullptr;
  void (*copy_to_initialized_indices)(const void *src, void *dst, IndexMask mask) = nullptr;

  void (*copy_to_uninitialized)(const void *src, void *dst) = nullptr;
  void (*copy_to_uninitialized_indices)(const void *src, void *dst, IndexMask mask) = nullptr;

  void (*move_to_initialized)(void *src, void *dst) = nullptr;
  void (*move_to_initialized_indices)(void *src, void *dst, IndexMask mask) = nullptr;

  void (*move_to_uninitialized)(void *src, void *dst) = nullptr;
  void (*move_to_uninitialized_indices)(void *src, void *dst, IndexMask mask) = nullptr;

  void (*relocate_to_initialized)(void *src, void *dst) = nullptr;
  void (*relocate_to_initialized_indices)(void *src, void *dst, IndexMask mask) = nullptr;

  void (*relocate_to_uninitialized)(void *src, void *dst) = nullptr;
  void (*relocate_to_uninitialized_indices)(void *src, void *dst, IndexMask mask) = nullptr;

  void (*fill_initialized)(const void *value, void *dst, int64_t n) = nullptr;
  void (*fill_initialized_indices)(const void *value, void *dst, IndexMask mask) = nullptr;

  void (*fill_uninitialized)(const void *value, void *dst, int64_t n) = nullptr;
  void (*fill_uninitialized_indices)(const void *value, void *dst, IndexMask mask) = nullptr;

  void (*debug_print)(const void *value, std::stringstream &ss) = nullptr;
  bool (*is_equal)(const void *a, const void *b) = nullptr;
  uint64_t (*hash)(const void *value) = nullptr;

  const void *default_value = nullptr;
  std::string name;
};

class CPPType : NonCopyable, NonMovable {
 private:
  CPPTypeMembers m_;

 public:
  CPPType(CPPTypeMembers members) : m_(std::move(members))
  {
    BLI_assert(is_power_of_2_i(m_.alignment));
    m_.alignment_mask = (uintptr_t)members.alignment - (uintptr_t)1;
    m_.has_special_member_functions = (m_.construct_default && m_.copy_to_uninitialized &&
                                       m_.copy_to_initialized && m_.move_to_uninitialized &&
                                       m_.move_to_initialized && m_.destruct);
  }

  /**
   * Two types only compare equal when their pointer is equal. No two instances of CPPType for the
   * same C++ type should be created.
   */
  friend bool operator==(const CPPType &a, const CPPType &b)
  {
    return &a == &b;
  }

  friend bool operator!=(const CPPType &a, const CPPType &b)
  {
    return !(&a == &b);
  }

  /**
   * Get the `CPPType` that corresponds to a specific static type.
   * This only works for types that actually implement the template specialization using
   * `MAKE_CPP_TYPE`.
   */
  template<typename T> static const CPPType &get();

  /**
   * Returns the name of the type for debugging purposes. This name should not be used as
   * identifier.
   */
  StringRefNull name() const
  {
    return m_.name;
  }

  /**
   * Required memory in bytes for an instance of this type.
   *
   * C++ equivalent:
   *   sizeof(T);
   */
  int64_t size() const
  {
    return m_.size;
  }

  /**
   * Required memory alignment for an instance of this type.
   *
   * C++ equivalent:
   *   alignof(T);
   */
  int64_t alignment() const
  {
    return m_.alignment;
  }

  /**
   * When true, the destructor does not have to be called on this type. This can sometimes be used
   * for optimization purposes.
   *
   * C++ equivalent:
   *   std::is_trivially_destructible_v<T>;
   */
  bool is_trivially_destructible() const
  {
    return m_.is_trivially_destructible;
  }

  bool is_default_constructible() const
  {
    return m_.construct_default != nullptr;
  }

  bool is_copy_constructible() const
  {
    return m_.copy_to_initialized != nullptr;
  }

  bool is_move_constructible() const
  {
    return m_.move_to_initialized != nullptr;
  }

  bool is_destructible() const
  {
    return m_.destruct != nullptr;
  }

  bool is_copy_assignable() const
  {
    return m_.copy_to_initialized != nullptr;
  }

  bool is_move_assignable() const
  {
    return m_.copy_to_uninitialized != nullptr;
  }

  /**
   * Returns true, when the type has the following functions:
   * - Default constructor.
   * - Copy constructor.
   * - Move constructor.
   * - Copy assignment operator.
   * - Move assignment operator.
   * - Destructor.
   */
  bool has_special_member_functions() const
  {
    return m_.has_special_member_functions;
  }

  /**
   * Returns true, when the given pointer fulfills the alignment requirement of this type.
   */
  bool pointer_has_valid_alignment(const void *ptr) const
  {
    return ((uintptr_t)ptr & m_.alignment_mask) == 0;
  }

  bool pointer_can_point_to_instance(const void *ptr) const
  {
    return ptr != nullptr && pointer_has_valid_alignment(ptr);
  }

  /**
   * Call the default constructor at the given memory location.
   * The memory should be uninitialized before this method is called.
   * For some trivial types (like int), this method does nothing.
   *
   * C++ equivalent:
   *   new (ptr) T;
   */
  void construct_default(void *ptr) const
  {
    BLI_assert(this->pointer_can_point_to_instance(ptr));

    m_.construct_default(ptr);
  }

  void construct_default_n(void *ptr, int64_t n) const
  {
    this->construct_default_indices(ptr, IndexMask(n));
  }

  void construct_default_indices(void *ptr, IndexMask mask) const
  {
    BLI_assert(mask.size() == 0 || this->pointer_can_point_to_instance(ptr));

    m_.construct_default_indices(ptr, mask);
  }

  /**
   * Call the destructor on the given instance of this type. The pointer must not be nullptr.
   *
   * For some trivial types, this does nothing.
   *
   * C++ equivalent:
   *   ptr->~T();
   */
  void destruct(void *ptr) const
  {
    BLI_assert(this->pointer_can_point_to_instance(ptr));

    m_.destruct(ptr);
  }

  void destruct_n(void *ptr, int64_t n) const
  {
    this->destruct_indices(ptr, IndexMask(n));
  }

  void destruct_indices(void *ptr, IndexMask mask) const
  {
    BLI_assert(mask.size() == 0 || this->pointer_can_point_to_instance(ptr));

    m_.destruct_indices(ptr, mask);
  }

  /**
   * Copy an instance of this type from src to dst.
   *
   * C++ equivalent:
   *   dst = src;
   */
  void copy_to_initialized(const void *src, void *dst) const
  {
    BLI_assert(src != dst);
    BLI_assert(this->pointer_can_point_to_instance(src));
    BLI_assert(this->pointer_can_point_to_instance(dst));

    m_.copy_to_initialized(src, dst);
  }

  void copy_to_initialized_n(const void *src, void *dst, int64_t n) const
  {
    this->copy_to_initialized_indices(src, dst, IndexMask(n));
  }

  void copy_to_initialized_indices(const void *src, void *dst, IndexMask mask) const
  {
    BLI_assert(mask.size() == 0 || src != dst);
    BLI_assert(mask.size() == 0 || this->pointer_can_point_to_instance(src));
    BLI_assert(mask.size() == 0 || this->pointer_can_point_to_instance(dst));

    m_.copy_to_initialized_indices(src, dst, mask);
  }

  /**
   * Copy an instance of this type from src to dst.
   *
   * The memory pointed to by dst should be uninitialized.
   *
   * C++ equivalent:
   *   new (dst) T(src);
   */
  void copy_to_uninitialized(const void *src, void *dst) const
  {
    BLI_assert(src != dst);
    BLI_assert(this->pointer_can_point_to_instance(src));
    BLI_assert(this->pointer_can_point_to_instance(dst));

    m_.copy_to_uninitialized(src, dst);
  }

  void copy_to_uninitialized_n(const void *src, void *dst, int64_t n) const
  {
    this->copy_to_uninitialized_indices(src, dst, IndexMask(n));
  }

  void copy_to_uninitialized_indices(const void *src, void *dst, IndexMask mask) const
  {
    BLI_assert(mask.size() == 0 || src != dst);
    BLI_assert(mask.size() == 0 || this->pointer_can_point_to_instance(src));
    BLI_assert(mask.size() == 0 || this->pointer_can_point_to_instance(dst));

    m_.copy_to_uninitialized_indices(src, dst, mask);
  }

  /**
   * Move an instance of this type from src to dst.
   *
   * The memory pointed to by dst should be initialized.
   *
   * C++ equivalent:
   *   dst = std::move(src);
   */
  void move_to_initialized(void *src, void *dst) const
  {
    BLI_assert(src != dst);
    BLI_assert(this->pointer_can_point_to_instance(src));
    BLI_assert(this->pointer_can_point_to_instance(dst));

    m_.move_to_initialized(src, dst);
  }

  void move_to_initialized_n(void *src, void *dst, int64_t n) const
  {
    this->move_to_initialized_indices(src, dst, IndexMask(n));
  }

  void move_to_initialized_indices(void *src, void *dst, IndexMask mask) const
  {
    BLI_assert(mask.size() == 0 || src != dst);
    BLI_assert(mask.size() == 0 || this->pointer_can_point_to_instance(src));
    BLI_assert(mask.size() == 0 || this->pointer_can_point_to_instance(dst));

    m_.move_to_initialized_indices(src, dst, mask);
  }

  /**
   * Move an instance of this type from src to dst.
   *
   * The memory pointed to by dst should be uninitialized.
   *
   * C++ equivalent:
   *   new (dst) T(std::move(src));
   */
  void move_to_uninitialized(void *src, void *dst) const
  {
    BLI_assert(src != dst);
    BLI_assert(this->pointer_can_point_to_instance(src));
    BLI_assert(this->pointer_can_point_to_instance(dst));

    m_.move_to_uninitialized(src, dst);
  }

  void move_to_uninitialized_n(void *src, void *dst, int64_t n) const
  {
    this->move_to_uninitialized_indices(src, dst, IndexMask(n));
  }

  void move_to_uninitialized_indices(void *src, void *dst, IndexMask mask) const
  {
    BLI_assert(mask.size() == 0 || src != dst);
    BLI_assert(mask.size() == 0 || this->pointer_can_point_to_instance(src));
    BLI_assert(mask.size() == 0 || this->pointer_can_point_to_instance(dst));

    m_.move_to_uninitialized_indices(src, dst, mask);
  }

  /**
   * Relocates an instance of this type from src to dst. src will point to uninitialized memory
   * afterwards.
   *
   * C++ equivalent:
   *   dst = std::move(src);
   *   src->~T();
   */
  void relocate_to_initialized(void *src, void *dst) const
  {
    BLI_assert(src != dst);
    BLI_assert(this->pointer_can_point_to_instance(src));
    BLI_assert(this->pointer_can_point_to_instance(dst));

    m_.relocate_to_initialized(src, dst);
  }

  void relocate_to_initialized_n(void *src, void *dst, int64_t n) const
  {
    this->relocate_to_initialized_indices(src, dst, IndexMask(n));
  }

  void relocate_to_initialized_indices(void *src, void *dst, IndexMask mask) const
  {
    BLI_assert(mask.size() == 0 || src != dst);
    BLI_assert(mask.size() == 0 || this->pointer_can_point_to_instance(src));
    BLI_assert(mask.size() == 0 || this->pointer_can_point_to_instance(dst));

    m_.relocate_to_initialized_indices(src, dst, mask);
  }

  /**
   * Relocates an instance of this type from src to dst. src will point to uninitialized memory
   * afterwards.
   *
   * C++ equivalent:
   *   new (dst) T(std::move(src))
   *   src->~T();
   */
  void relocate_to_uninitialized(void *src, void *dst) const
  {
    BLI_assert(src != dst);
    BLI_assert(this->pointer_can_point_to_instance(src));
    BLI_assert(this->pointer_can_point_to_instance(dst));

    m_.relocate_to_uninitialized(src, dst);
  }

  void relocate_to_uninitialized_n(void *src, void *dst, int64_t n) const
  {
    this->relocate_to_uninitialized_indices(src, dst, IndexMask(n));
  }

  void relocate_to_uninitialized_indices(void *src, void *dst, IndexMask mask) const
  {
    BLI_assert(mask.size() == 0 || src != dst);
    BLI_assert(mask.size() == 0 || this->pointer_can_point_to_instance(src));
    BLI_assert(mask.size() == 0 || this->pointer_can_point_to_instance(dst));

    m_.relocate_to_uninitialized_indices(src, dst, mask);
  }

  /**
   * Copy the given value to the first n elements in an array starting at dst.
   *
   * Other instances of the same type should live in the array before this method is called.
   */
  void fill_initialized(const void *value, void *dst, int64_t n) const
  {
    BLI_assert(n == 0 || this->pointer_can_point_to_instance(value));
    BLI_assert(n == 0 || this->pointer_can_point_to_instance(dst));

    m_.fill_initialized(value, dst, n);
  }

  void fill_initialized_indices(const void *value, void *dst, IndexMask mask) const
  {
    BLI_assert(mask.size() == 0 || this->pointer_can_point_to_instance(value));
    BLI_assert(mask.size() == 0 || this->pointer_can_point_to_instance(dst));

    m_.fill_initialized_indices(value, dst, mask);
  }

  /**
   * Copy the given value to the first n elements in an array starting at dst.
   *
   * The array should be uninitialized before this method is called.
   */
  void fill_uninitialized(const void *value, void *dst, int64_t n) const
  {
    BLI_assert(n == 0 || this->pointer_can_point_to_instance(value));
    BLI_assert(n == 0 || this->pointer_can_point_to_instance(dst));

    m_.fill_uninitialized(value, dst, n);
  }

  void fill_uninitialized_indices(const void *value, void *dst, IndexMask mask) const
  {
    BLI_assert(mask.size() == 0 || this->pointer_can_point_to_instance(value));
    BLI_assert(mask.size() == 0 || this->pointer_can_point_to_instance(dst));

    m_.fill_uninitialized_indices(value, dst, mask);
  }

  void debug_print(const void *value, std::stringstream &ss) const
  {
    BLI_assert(this->pointer_can_point_to_instance(value));
    m_.debug_print(value, ss);
  }

  bool is_equal(const void *a, const void *b) const
  {
    BLI_assert(this->pointer_can_point_to_instance(a));
    BLI_assert(this->pointer_can_point_to_instance(b));
    return m_.is_equal(a, b);
  }

  uint64_t hash(const void *value) const
  {
    BLI_assert(this->pointer_can_point_to_instance(value));
    return m_.hash(value);
  }

  /**
   * Get a pointer to a constant value of this type. The specific value depends on the type.
   * It is usually a zero-initialized or default constructed value.
   */
  const void *default_value() const
  {
    return m_.default_value;
  }

  uint64_t hash() const
  {
    return get_default_hash(this);
  }

  /**
   * Low level access to the callbacks for this CPPType.
   */
  const CPPTypeMembers &members() const
  {
    return m_;
  }

  template<typename T> bool is() const
  {
    return this == &CPPType::get<std::decay_t<T>>();
  }
};

}  // namespace blender::fn

/* Utility for allocating an uninitialized buffer for a single value of the given #CPPType. */
#define BUFFER_FOR_CPP_TYPE_VALUE(type, variable_name) \
  blender::DynamicStackBuffer<64, 64> stack_buffer_for_##variable_name((type).size(), \
                                                                       (type).alignment()); \
  void *variable_name = stack_buffer_for_##variable_name.buffer();
