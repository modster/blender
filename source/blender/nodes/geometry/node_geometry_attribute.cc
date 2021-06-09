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

#include "node_geometry_attribute.hh"

#include "FN_cpp_type_make.hh"

const AttributeRef AttributeRef::None = AttributeRef("", ATTR_DOMAIN_AUTO);

AttributeRef::AttributeRef()
{
}

AttributeRef::AttributeRef(const std::string &name, AttributeDomain domain)
    : name_(name), domain_(domain)
{
}

std::ostream &operator<<(std::ostream &stream, const AttributeRef &attr)
{
  stream << "<AttributeRef name=" << attr.name_ << ", domain=" << attr.domain_ << ">";
  return stream;
}

/* This generally should not be used. It is necessary currently, so that GeometrySet can by used by
 * the CPPType system. */
bool operator==(const AttributeRef &UNUSED(a), const AttributeRef &UNUSED(b))
{
  return false;
}

/* This generally should not be used. It is necessary currently, so that GeometrySet can by used by
 * the CPPType system. */
uint64_t AttributeRef::hash() const
{
  return reinterpret_cast<uint64_t>(this);
}

MAKE_CPP_TYPE(AttributeRef, AttributeRef);
