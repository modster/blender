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

#include <string>

#include "BLI_color.hh"
#include "BLI_float3.hh"

#include "FN_cpp_type.hh"

#include "BKE_attribute.h"

/**
 * Runtime data struct that references attributes during evaluation of geometry node trees.
 * Attributes are identified by a name and a domain type.
 */
struct AttributeRef {
 private:
  std::string name_;
  CustomDataType data_type_;

  /* Single value to use when the socket is not connected. */
  union {
    float value_float_;
    int value_int_;
    bool value_bool_;
    blender::float3 value_float3_;
    blender::ColorGeometry4f value_color_;
  };

 public:
  static const AttributeRef None;

 public:
  const std::string &name() const;
  CustomDataType data_type() const;

  AttributeRef();
  AttributeRef(CustomDataType data_type);
  AttributeRef(const std::string &name, CustomDataType data_type);

  friend std::ostream &operator<<(std::ostream &stream, const AttributeRef &geometry_set);
  friend bool operator==(const AttributeRef &a, const AttributeRef &b);
  uint64_t hash() const;

  bool valid() const;

  void *single_value_ptr();
  const void *single_value_ptr() const;

  template<typename T> T &single_value()
  {
    BLI_assert(blender::fn::CPPType::get<T>() ==
               *blender::bke::custom_data_type_to_cpp_type(data_type_));
    return *(T *)single_value_ptr();
  }
  template<typename T> const T &single_value() const
  {
    BLI_assert(blender::fn::CPPType::get<T>() ==
               *blender::bke::custom_data_type_to_cpp_type(data_type_));
    return *(const T *)single_value_ptr();
  }
};
