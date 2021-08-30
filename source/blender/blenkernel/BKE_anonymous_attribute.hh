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

#include <atomic>
#include <string>

#include "BLI_string_ref.hh"

#include "BKE_anonymous_attribute.h"

namespace blender::bke {

template<bool IsStrongReference> class OwnedAnonymousAttributeID {
 private:
  const AnonymousAttributeID *data_ = nullptr;

 public:
  OwnedAnonymousAttributeID() = default;

  explicit OwnedAnonymousAttributeID(StringRefNull debug_name)
  {
    if constexpr (IsStrongReference) {
      data_ = BKE_anonymous_attribute_id_new_strong(debug_name.c_str());
    }
    else {
      data_ = BKE_anonymous_attribute_id_new_weak(debug_name.c_str());
    }
  }

  /* This transfers ownership, so no incref is necessary. */
  explicit OwnedAnonymousAttributeID(const AnonymousAttributeID *anonymous_id)
      : data_(anonymous_id)
  {
  }

  template<bool OtherIsStrong>
  OwnedAnonymousAttributeID(const OwnedAnonymousAttributeID<OtherIsStrong> &other)
  {
    data_ = other.data_;
    this->incref();
  }

  template<bool OtherIsStrong>
  OwnedAnonymousAttributeID(OwnedAnonymousAttributeID<OtherIsStrong> &&other)
  {
    data_ = other.data_;
    this->incref();
    other.decref();
    other.data_ = nullptr;
  }

  ~OwnedAnonymousAttributeID()
  {
    this->decref();
  }

  template<bool OtherIsStrong>
  OwnedAnonymousAttributeID &operator=(const OwnedAnonymousAttributeID<OtherIsStrong> &other)
  {
    if (this == &other) {
      return *this;
    }
    this->~OwnedAnonymousAttributeID();
    new (this) OwnedAnonymousAttributeID(other);
    return *this;
  }

  template<bool OtherIsStrong>
  OwnedAnonymousAttributeID &operator=(OwnedAnonymousAttributeID<OtherIsStrong> &&other)
  {
    if (this == &other) {
      return *this;
    }
    this->~OwnedAnonymousAttributeID();
    new (this) OwnedAnonymousAttributeID(std::move(other));
    return *this;
  }

  operator bool() const
  {
    return data_ != nullptr;
  }

  StringRefNull debug_name() const
  {
    BLI_assert(data_ != nullptr);
    return BKE_anonymous_attribute_id_debug_name(data_);
  }

  bool has_strong_references() const
  {
    BLI_assert(data_ != nullptr);
    return BKE_anonymous_attribute_id_has_strong_references(data_);
  }

  const AnonymousAttributeID *extract()
  {
    const AnonymousAttributeID *extracted_data = data_;
    /* Don't decref because the caller becomes the new owner. */
    data_ = nullptr;
    return extracted_data;
  }

  const AnonymousAttributeID *get()
  {
    return data_;
  }

 private:
  void incref()
  {
    if (data_ == nullptr) {
      return;
    }
    if constexpr (IsStrongReference) {
      BKE_anonymous_attribute_id_increment_strong(data_);
    }
    else {
      BKE_anonymous_attribute_id_increment_weak(data_);
    }
  }

  void decref()
  {
    if (data_ == nullptr) {
      return;
    }
    if constexpr (IsStrongReference) {
      BKE_anonymous_attribute_id_decrement_strong(data_);
    }
    else {
      BKE_anonymous_attribute_id_decrement_weak(data_);
    }
  }
};

using StrongAnonymousAttributeID = OwnedAnonymousAttributeID<true>;
using WeakAnonymousAttributeID = OwnedAnonymousAttributeID<false>;

class AttributeIDRef {
 private:
  StringRef name_;
  const AnonymousAttributeID *anonymous_id_ = nullptr;

 public:
  AttributeIDRef() = default;

  AttributeIDRef(StringRef name) : name_(name)
  {
  }

  AttributeIDRef(StringRefNull name) : name_(name)
  {
  }

  AttributeIDRef(const char *name) : name_(name)
  {
  }

  AttributeIDRef(const std::string &name) : name_(name)
  {
  }

  /* The anonymous id is only borrowed, the caller has to keep a reference to it. */
  AttributeIDRef(const AnonymousAttributeID *anonymous_id) : anonymous_id_(anonymous_id)
  {
  }

  operator bool() const
  {
    return this->is_named() || this->is_anonymous();
  }

  bool is_named() const
  {
    return !name_.is_empty();
  }

  bool is_anonymous() const
  {
    return anonymous_id_ != nullptr;
  }

  StringRef name() const
  {
    BLI_assert(this->is_named());
    return name_;
  }

  const AnonymousAttributeID &anonymous_id() const
  {
    BLI_assert(this->is_anonymous());
    return *anonymous_id_;
  }
};

}  // namespace blender::bke
