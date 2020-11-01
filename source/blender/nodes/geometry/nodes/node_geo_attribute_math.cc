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

#include "DNA_windowmanager_types.h"

#include "BKE_attribute.h"

#include "node_geometry_util.hh"

static bNodeSocketTemplate geo_node_attribute_math_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_STRING, N_("Attribute A")},
    {SOCK_STRING, N_("Attribute B")},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_attribute_math_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    // {SOCK_STRING, N_("Attribute")},
    {-1, ""},
};

namespace blender::nodes {
static void geo_attribute_math_exec(bNode *UNUSED(node),
                                    GeoNodeInputs inputs,
                                    GeoNodeOutputs outputs)
{
  GeometryPtr geometry = inputs.extract<GeometryPtr>("Geometry");

  if (!geometry.has_value()) {
    outputs.set("Geometry", std::move(geometry));
    return;
  }

  make_geometry_mutable(geometry);

  Mesh *mesh = geometry->get_mesh_for_write();
  std::string attribute_name_a = inputs.extract<std::string>("Attribute A");
  std::string attribute_name_b = inputs.extract<std::string>("Attribute B");

  outputs.set("Geometry", std::move(geometry));
}
}  // namespace blender::nodes

void register_node_type_geo_attribute_math()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_ATTRIBUTE_MATH, "Attribute Math", 0, 0);
  node_type_socket_templates(&ntype, geo_node_attribute_math_in, geo_node_attribute_math_out);
  ntype.geometry_node_execute = blender::nodes::geo_attribute_math_exec;
  nodeRegisterType(&ntype);
}
