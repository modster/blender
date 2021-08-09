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

static bNodeSocketTemplate geo_node_attribute_store_local_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_VECTOR, N_("Value"), 0.0f, 0.0f, 0.0f, 0.0f, -FLT_MAX, FLT_MAX},
    {SOCK_FLOAT, N_("Value"), 0.0f, 0.0f, 0.0f, 0.0f, -FLT_MAX, FLT_MAX},
    {SOCK_RGBA, N_("Value"), 0.0f, 0.0f, 0.0f, 0.0f, -FLT_MAX, FLT_MAX},
    {SOCK_BOOLEAN, N_("Value"), 0.0f, 0.0f, 0.0f, 0.0f, -FLT_MAX, FLT_MAX},
    {SOCK_INT, N_("Value"), 0, 0, 0, 0, -10000000.0f, 10000000.0f},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_attribute_store_local_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

static void geo_node_attribute_store_local_layout(uiLayout *layout,
                                                  bContext *UNUSED(C),
                                                  PointerRNA *ptr)
{
  uiLayoutSetPropSep(layout, true);
  uiLayoutSetPropDecorate(layout, false);
  uiItemR(layout, ptr, "domain", 0, "", ICON_NONE);
  uiItemR(layout, ptr, "data_type", 0, "", ICON_NONE);
}

static void geo_node_attribute_store_local_init(bNodeTree *UNUSED(tree), bNode *node)
{
  node->custom1 = CD_PROP_FLOAT;
  node->custom2 = ATTR_DOMAIN_POINT;
}

static void geo_node_attribute_store_local_update(bNodeTree *UNUSED(ntree), bNode *node)
{
  bNodeSocket *socket_value_vector = (bNodeSocket *)BLI_findlink(&node->inputs, 1);
  bNodeSocket *socket_value_float = socket_value_vector->next;
  bNodeSocket *socket_value_color4f = socket_value_float->next;
  bNodeSocket *socket_value_boolean = socket_value_color4f->next;
  bNodeSocket *socket_value_int32 = socket_value_boolean->next;

  const CustomDataType data_type = static_cast<CustomDataType>(node->custom1);

  nodeSetSocketAvailability(socket_value_vector, data_type == CD_PROP_FLOAT3);
  nodeSetSocketAvailability(socket_value_float, data_type == CD_PROP_FLOAT);
  nodeSetSocketAvailability(socket_value_color4f, data_type == CD_PROP_COLOR);
  nodeSetSocketAvailability(socket_value_boolean, data_type == CD_PROP_BOOL);
  nodeSetSocketAvailability(socket_value_int32, data_type == CD_PROP_INT32);
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

static void store_local_attribute(GeometryComponent &component, const GeoNodeExecParams &params)
{
  const bNode &node = params.node();
  const bNodeTree &tree = params.ntree();
  const std::string attribute_name = get_local_attribute_name(tree.id.name, node.name, "");

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
      const Array<float> values = params.get_input<Array<float>>("Value_001");
      fn::GVArray_For_RepeatedGSpan values_repeated{attribute->size(), values.as_span()};
      fn::GVArray_GSpan values_span{values_repeated};
      attribute->set_all(values_span.data());
      break;
    }
    case CD_PROP_FLOAT3: {
      const Array<float3> values = params.get_input<Array<float3>>("Value");
      fn::GVArray_For_RepeatedGSpan values_repeated{attribute->size(), values.as_span()};
      fn::GVArray_GSpan values_span{values_repeated};
      attribute->set_all(values_span.data());
      break;
    }
    case CD_PROP_COLOR: {
      const Array<ColorGeometry4f> values = params.get_input<Array<ColorGeometry4f>>("Value_002");
      fn::GVArray_For_RepeatedGSpan values_repeated{attribute->size(), values.as_span()};
      fn::GVArray_GSpan values_span{values_repeated};
      attribute->set_all(values_span.data());
      break;
    }
    case CD_PROP_BOOL: {
      const Array<bool> values = params.get_input<Array<bool>>("Value_003");
      fn::GVArray_For_RepeatedGSpan values_repeated{attribute->size(), values.as_span()};
      fn::GVArray_GSpan values_span{values_repeated};
      attribute->set_all(values_span.data());
      break;
    }
    case CD_PROP_INT32: {
      const Array<int> values = params.get_input<Array<int>>("Value_004");
      fn::GVArray_For_RepeatedGSpan values_repeated{attribute->size(), values.as_span()};
      fn::GVArray_GSpan values_span{values_repeated};
      attribute->set_all(values_span.data());
      break;
    }
    default:
      break;
  }

  attribute.save();
}

static void geo_node_attribute_store_local_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");

  geometry_set = geometry_set_realize_instances(geometry_set);

  if (geometry_set.has<MeshComponent>()) {
    store_local_attribute(geometry_set.get_component_for_write<MeshComponent>(), params);
  }
  if (geometry_set.has<PointCloudComponent>()) {
    store_local_attribute(geometry_set.get_component_for_write<PointCloudComponent>(), params);
  }
  if (geometry_set.has<CurveComponent>()) {
    store_local_attribute(geometry_set.get_component_for_write<CurveComponent>(), params);
  }

  params.set_output("Geometry", geometry_set);
}

}  // namespace blender::nodes

void register_node_type_geo_attribute_store_local()
{
  static bNodeType ntype;

  geo_node_type_base(
      &ntype, GEO_NODE_ATTRIBUTE_STORE_LOCAL, "Store Local Attribute", NODE_CLASS_ATTRIBUTE, 0);
  node_type_socket_templates(
      &ntype, geo_node_attribute_store_local_in, geo_node_attribute_store_local_out);
  node_type_init(&ntype, geo_node_attribute_store_local_init);
  node_type_update(&ntype, geo_node_attribute_store_local_update);
  ntype.geometry_node_execute = blender::nodes::geo_node_attribute_store_local_exec;
  ntype.draw_buttons = geo_node_attribute_store_local_layout;
  nodeRegisterType(&ntype);
}
