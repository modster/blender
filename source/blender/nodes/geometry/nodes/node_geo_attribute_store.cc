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

#include "UI_interface.h"
#include "UI_resources.h"

#include "node_geometry_util.hh"

static bNodeSocketTemplate geo_node_attribute_store_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_ATTRIBUTE, N_("Attribute")},
    {SOCK_STRING, N_("Name")},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_attribute_store_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

static void geo_node_attribute_store_init(bNodeTree *UNUSED(tree), bNode *node)
{
  nodeFindSocket(node, SOCK_IN, "Attribute")->display_shape = SOCK_DISPLAY_SHAPE_SQUARE;
}

namespace blender::nodes {

static AttributeDomain get_result_domain(AttributeDomain declared_domain,
                                         const GeometryComponent &component,
                                         const StringRef name)
{
  if (declared_domain == ATTR_DOMAIN_AUTO) {
    /* Use the domain of the result attribute if it already exists. */
    std::optional<AttributeMetaData> result_info = component.attribute_get_meta_data(name);
    if (result_info) {
      return result_info->domain;
    }
    else {
      return ATTR_DOMAIN_POINT;
    }
  }
  else {
    return declared_domain;
  }
}

//static void store_attribute(GeometryComponent &component, const GeoNodeExecParams &params)
//{
//  const AttributeRef &attribute_ref = params.get_input<AttributeRef>("Attribute");
//  const CustomDataType data_type = attribute_ref.data_type();
//  const AttributeDomain domain = get_result_domain(attribute_ref.domain(), component, input_name);
//  const std::string attribute_name = params.get_input<std::string>("Name");
//
//  GVArrayPtr attribute_input = component.attribute_try_get_for_read(
//      attribute_ref.name(), domain, data_type);
//  if (!attribute_input) {
//    return;
//  }
//  OutputAttribute attribute_output = component.attribute_try_get_for_output_only(
//      attribute_name, domain, data_type);
//
//  MutableSpan<float> results = attribute_output.as_span<float>();
//  results.copy_from(attribute_input->typed<float>());
//
//  switch (data_type) {
//    case CD_PROP_FLOAT: {
//      const float value = params.get_input<float>("Attribute");
//      attribute->fill(&value);
//      break;
//    }
//    case CD_PROP_FLOAT3: {
//      const float3 value = params.get_input<float3>("Attribute");
//      attribute->fill(&value);
//      break;
//    }
//    case CD_PROP_COLOR: {
//      const ColorGeometry4f value = params.get_input<ColorGeometry4f>("Attribute");
//      attribute->fill(&value);
//      break;
//    }
//    case CD_PROP_BOOL: {
//      const bool value = params.get_input<bool>("Attribute");
//      attribute->fill(&value);
//      break;
//    }
//    case CD_PROP_INT32: {
//      const int value = params.get_input<int>("Attribute");
//      attribute->fill(&value);
//      break;
//    }
//    default:
//      break;
//  }
//
//  attribute.save();
//}

static void geo_node_attribute_store_exec(GeoNodeExecParams params)
{
  params.set_output("Geometry", params.extract_input<GeometrySet>("Geometry"));

  //GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");

  //const std::string attribute_name = params.get_input<std::string>("Name");
  //if (attribute_name.empty()) {
  //  params.set_output("Geometry", geometry_set);
  //  return;
  //}

  //// geometry_set = geometry_set_realize_instances(geometry_set);

  //if (geometry_set.has<MeshComponent>()) {
  //  store_attribute(geometry_set.get_component_for_write<MeshComponent>(), params);
  //}
  //if (geometry_set.has<PointCloudComponent>()) {
  //  store_attribute(geometry_set.get_component_for_write<PointCloudComponent>(), params);
  //}
  //if (geometry_set.has<CurveComponent>()) {
  //  store_attribute(geometry_set.get_component_for_write<CurveComponent>(), params);
  //}

  //params.set_output("Geometry", geometry_set);
}

}  // namespace blender::nodes

void register_node_type_geo_attribute_store()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_ATTRIBUTE_STORE, "Attribute Store", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, geo_node_attribute_store_in, geo_node_attribute_store_out);
  node_type_init(&ntype, geo_node_attribute_store_init);
  ntype.geometry_node_execute = blender::nodes::geo_node_attribute_store_exec;
  nodeRegisterType(&ntype);
}
