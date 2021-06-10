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

static void geo_node_attribute_store_layout(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
{
  uiLayoutSetPropSep(layout, true);
  uiLayoutSetPropDecorate(layout, false);
  uiItemR(layout, ptr, "domain", 0, "", ICON_NONE);
  uiItemR(layout, ptr, "data_type", 0, "", ICON_NONE);
}

namespace blender::nodes {

static AttributeDomain get_result_domain(const GeometryComponent &component, const StringRef name)
{
  /* Use the domain of the result attribute if it already exists. */
  std::optional<AttributeMetaData> result_info = component.attribute_get_meta_data(name);
  if (result_info) {
    return result_info->domain;
  }
  return ATTR_DOMAIN_POINT;
}

static void store_attribute(GeometryComponent &component, const GeoNodeExecParams &params)
{
  const std::string attribute_name = params.get_input<std::string>("Name");
  if (attribute_name.empty()) {
    return;
  }

  const bNode &node = params.node();
  const CustomDataType data_type = static_cast<CustomDataType>(node.custom1);
  const AttributeDomain domain = static_cast<AttributeDomain>(node.custom2);
  const AttributeDomain result_domain = (domain == ATTR_DOMAIN_AUTO) ?
                                            get_result_domain(component, attribute_name) :
                                            domain;

  OutputAttribute attribute = component.attribute_try_get_for_output_only(
      attribute_name, result_domain, data_type);
  if (!attribute) {
    return;
  }

  switch (data_type) {
    case CD_PROP_FLOAT: {
      const float value = params.get_input<float>("Attribute");
      attribute->fill(&value);
      break;
    }
    case CD_PROP_FLOAT3: {
      const float3 value = params.get_input<float3>("Attribute");
      attribute->fill(&value);
      break;
    }
    case CD_PROP_COLOR: {
      const ColorGeometry4f value = params.get_input<ColorGeometry4f>("Attribute");
      attribute->fill(&value);
      break;
    }
    case CD_PROP_BOOL: {
      const bool value = params.get_input<bool>("Attribute");
      attribute->fill(&value);
      break;
    }
    case CD_PROP_INT32: {
      const int value = params.get_input<int>("Attribute");
      attribute->fill(&value);
      break;
    }
    default:
      break;
  }

  attribute.save();
}

static void geo_node_attribute_store_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");

  geometry_set = geometry_set_realize_instances(geometry_set);

  if (geometry_set.has<MeshComponent>()) {
    store_attribute(geometry_set.get_component_for_write<MeshComponent>(), params);
  }
  if (geometry_set.has<PointCloudComponent>()) {
    store_attribute(geometry_set.get_component_for_write<PointCloudComponent>(), params);
  }
  if (geometry_set.has<CurveComponent>()) {
    store_attribute(geometry_set.get_component_for_write<CurveComponent>(), params);
  }

  params.set_output("Geometry", geometry_set);
}

}  // namespace blender::nodes

void register_node_type_geo_attribute_store()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_ATTRIBUTE_STORE, "Attribute Store", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, geo_node_attribute_store_in, geo_node_attribute_store_out);
  ntype.geometry_node_execute = blender::nodes::geo_node_attribute_store_exec;
  ntype.draw_buttons = geo_node_attribute_store_layout;
  nodeRegisterType(&ntype);
}
