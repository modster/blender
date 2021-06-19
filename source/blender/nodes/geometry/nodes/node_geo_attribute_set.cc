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

#include "BLI_task.hh"

#include "UI_interface.h"
#include "UI_resources.h"

#include "node_geometry_util.hh"

static bNodeSocketTemplate geo_node_attribute_set_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_ATTRIBUTE, N_("Attribute")},
    {SOCK_STRING, N_("Name")},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_attribute_set_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

static void geo_node_attribute_set_init(bNodeTree *UNUSED(tree), bNode *node)
{
  blender::nodes::set_attribute_socket_data_type(*node, "Attribute", SOCK_ATTRIBUTE);
}

namespace blender::nodes {

static void set_attribute(GeometryComponent &component,
                          const GeoNodeExecParams &params,
                          const AttributeRef &attribute_ref,
                          const StringRef attribute_name)
{
  ReadAttributeLookup attribute_input = component.attribute_try_get_for_read(
      attribute_ref.name(), attribute_ref.data_type());

  if (attribute_input) {
    OutputAttribute attribute_output = component.attribute_try_get_for_output_only(
        attribute_name, attribute_input.domain, attribute_ref.data_type());

    if (attribute_output) {
      threading::parallel_for(IndexRange(attribute_output->size()), 512, [&](IndexRange range) {
        BUFFER_FOR_CPP_TYPE_VALUE(attribute_output.cpp_type(), buffer);
        for (const int i : range) {
          attribute_input.varray->get(i, buffer);
          attribute_output->set_by_relocate(i, buffer);
        }
      });
    }

    attribute_output.save();
  }
}

static void geo_node_attribute_set_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");
  const AttributeRef &attribute_ref = params.extract_input<AttributeRef>("Attribute");
  const std::string attribute_name = params.extract_input<std::string>("Name");

  if (attribute_name.empty()) {
    params.set_output("Geometry", geometry_set);
    return;
  }

  for (const GeometryComponentType component_type : {GEO_COMPONENT_TYPE_MESH,
                                                     GEO_COMPONENT_TYPE_POINT_CLOUD,
                                                     GEO_COMPONENT_TYPE_INSTANCES,
                                                     GEO_COMPONENT_TYPE_VOLUME,
                                                     GEO_COMPONENT_TYPE_CURVE}) {
    if (geometry_set.has(component_type)) {
      set_attribute(geometry_set.get_component_for_write(component_type),
                    params,
                    attribute_ref,
                    attribute_name);
    }
  }

  params.set_output("Geometry", geometry_set);
}

}  // namespace blender::nodes

void register_node_type_geo_attribute_set()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_ATTRIBUTE_SET, "Attribute Set", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, geo_node_attribute_set_in, geo_node_attribute_set_out);
  node_type_init(&ntype, geo_node_attribute_set_init);
  ntype.geometry_node_execute = blender::nodes::geo_node_attribute_set_exec;
  nodeRegisterType(&ntype);
}
