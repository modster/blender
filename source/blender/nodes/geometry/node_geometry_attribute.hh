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

#include "BKE_attribute.h"

/**
 * Runtime data struct that references attributes during evaluation of geometry node trees.
 * Attributes are identified by a name and a domain type.
 */
struct AttributeRef {
 private:
  std::string name_;
  AttributeDomain domain_;

 public:
  static const AttributeRef None;

 public:
  AttributeRef();
  AttributeRef(const std::string &name, AttributeDomain domain);

  friend std::ostream &operator<<(std::ostream &stream, const AttributeRef &geometry_set);
  friend bool operator==(const AttributeRef &a, const AttributeRef &b);
  uint64_t hash() const;
};
