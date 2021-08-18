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

#include "BKE_attribute_math.hh"

#include "node_geometry_util.hh"

static bNodeSocketTemplate geo_node_attribute_freeze_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_VECTOR, N_("Value"), 0.0f, 0.0f, 0.0f, 0.0f, -FLT_MAX, FLT_MAX, PROP_NONE, SOCK_FIELD},
    {SOCK_FLOAT, N_("Value"), 0.0f, 0.0f, 0.0f, 0.0f, -FLT_MAX, FLT_MAX, PROP_NONE, SOCK_FIELD},
    {SOCK_RGBA, N_("Value"), 0.0f, 0.0f, 0.0f, 0.0f, -FLT_MAX, FLT_MAX, PROP_NONE, SOCK_FIELD},
    {SOCK_BOOLEAN, N_("Value"), 0.0f, 0.0f, 0.0f, 0.0f, -FLT_MAX, FLT_MAX, PROP_NONE, SOCK_FIELD},
    {SOCK_INT, N_("Value"), 0, 0, 0, 0, -10000000.0f, 10000000.0f, PROP_NONE, SOCK_FIELD},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_attribute_freeze_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_VECTOR, N_("Attribute"), 0.0f, 0.0f, 0.0f, 0.0f, -FLT_MAX, FLT_MAX},
    {SOCK_FLOAT, N_("Attribute"), 0.0f, 0.0f, 0.0f, 0.0f, -FLT_MAX, FLT_MAX},
    {SOCK_RGBA, N_("Attribute"), 0.0f, 0.0f, 0.0f, 0.0f, -FLT_MAX, FLT_MAX},
    {SOCK_BOOLEAN, N_("Attribute"), 0.0f, 0.0f, 0.0f, 0.0f, -FLT_MAX, FLT_MAX},
    {SOCK_INT, N_("Attribute"), 0, 0, 0, 0, -10000000.0f, 10000000.0f},
    {-1, ""},
};

static void geo_node_attribute_freeze_layout(uiLayout *layout,
                                             bContext *UNUSED(C),
                                             PointerRNA *ptr)
{
  uiLayoutSetPropSep(layout, true);
  uiLayoutSetPropDecorate(layout, false);
  uiItemR(layout, ptr, "domain", 0, "", ICON_NONE);
  uiItemR(layout, ptr, "data_type", 0, "", ICON_NONE);
}

static void geo_node_attribute_freeze_init(bNodeTree *UNUSED(tree), bNode *node)
{
  NodeGeometryAttributeFreeze *data = (NodeGeometryAttributeFreeze *)MEM_callocN(
      sizeof(NodeGeometryAttributeFreeze), __func__);
  data->data_type = CD_PROP_FLOAT;
  data->domain = ATTR_DOMAIN_POINT;

  node->storage = data;
}

static void geo_node_attribute_freeze_update(bNodeTree *UNUSED(ntree), bNode *node)
{
  const NodeGeometryAttributeFreeze &storage = *(const NodeGeometryAttributeFreeze *)node->storage;
  const CustomDataType data_type = static_cast<CustomDataType>(storage.data_type);

  bNodeSocket *socket_value_attribute_name = (bNodeSocket *)node->inputs.first;
  bNodeSocket *socket_value_vector = socket_value_attribute_name->next;
  bNodeSocket *socket_value_float = socket_value_vector->next;
  bNodeSocket *socket_value_color4f = socket_value_float->next;
  bNodeSocket *socket_value_boolean = socket_value_color4f->next;
  bNodeSocket *socket_value_int32 = socket_value_boolean->next;

  nodeSetSocketAvailability(socket_value_vector, data_type == CD_PROP_FLOAT3);
  nodeSetSocketAvailability(socket_value_float, data_type == CD_PROP_FLOAT);
  nodeSetSocketAvailability(socket_value_color4f, data_type == CD_PROP_COLOR);
  nodeSetSocketAvailability(socket_value_boolean, data_type == CD_PROP_BOOL);
  nodeSetSocketAvailability(socket_value_int32, data_type == CD_PROP_INT32);

  bNodeSocket *out_socket_value_attribute_name = (bNodeSocket *)node->outputs.first;
  bNodeSocket *out_socket_value_vector = out_socket_value_attribute_name->next;
  bNodeSocket *out_socket_value_float = out_socket_value_vector->next;
  bNodeSocket *out_socket_value_color4f = out_socket_value_float->next;
  bNodeSocket *out_socket_value_boolean = out_socket_value_color4f->next;
  bNodeSocket *out_socket_value_int32 = out_socket_value_boolean->next;

  nodeSetSocketAvailability(out_socket_value_vector, data_type == CD_PROP_FLOAT3);
  nodeSetSocketAvailability(out_socket_value_float, data_type == CD_PROP_FLOAT);
  nodeSetSocketAvailability(out_socket_value_color4f, data_type == CD_PROP_COLOR);
  nodeSetSocketAvailability(out_socket_value_boolean, data_type == CD_PROP_BOOL);
  nodeSetSocketAvailability(out_socket_value_int32, data_type == CD_PROP_INT32);
}

namespace blender::nodes {

template<typename T>
void set_output_field(GeoNodeExecParams &params,
                      AnonymousCustomDataLayerID &layer_id,
                      const StringRef output_name)
{
  params.set_output(
      output_name,
      bke::FieldRef<T>(new bke::AnonymousAttributeField(layer_id, CPPType::get<T>())));
}

static void set_output_field(GeoNodeExecParams &params,
                             AnonymousCustomDataLayerID &layer_id,
                             const CustomDataType data_type)
{
  switch (data_type) {
    case CD_PROP_FLOAT: {
      set_output_field<float>(params, layer_id, "Attribute_001");
      break;
    }
    case CD_PROP_FLOAT3: {
      set_output_field<float3>(params, layer_id, "Attribute");
      break;
    }
    case CD_PROP_COLOR: {
      set_output_field<ColorGeometry4f>(params, layer_id, "Attribute_002");
      break;
    }
    case CD_PROP_BOOL: {
      set_output_field<bool>(params, layer_id, "Attribute_003");
      break;
    }
    case CD_PROP_INT32: {
      set_output_field<int>(params, layer_id, "Attribute_004");
      break;
    }
    default:
      break;
  }
}

static void geo_node_attribute_freeze_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");

  geometry_set = bke::geometry_set_realize_instances(geometry_set);

  const bNode &node = params.node();
  const NodeGeometryAttributeFreeze &storage = *(const NodeGeometryAttributeFreeze *)node.storage;
  const CustomDataType data_type = static_cast<CustomDataType>(storage.data_type);
  const AttributeDomain domain = static_cast<AttributeDomain>(storage.domain);

  AnonymousCustomDataLayerID *id = CustomData_anonymous_id_new("Attribute Freeze");

  FieldPtr field;
  switch (data_type) {
    case CD_PROP_FLOAT:
      field = params.get_input_field<float>("Value_001").field();
      break;
    case CD_PROP_FLOAT3:
      field = params.get_input_field<float3>("Value").field();
      break;
    case CD_PROP_COLOR:
      field = params.get_input_field<ColorGeometry4f>("Value_002").field();
      break;
    case CD_PROP_BOOL:
      field = params.get_input_field<bool>("Value_003").field();
      break;
    case CD_PROP_INT32:
      field = params.get_input_field<int>("Value_004").field();
      break;
    default:
      break;
  }

  static const Array<GeometryComponentType> types = {
      GEO_COMPONENT_TYPE_MESH, GEO_COMPONENT_TYPE_POINT_CLOUD, GEO_COMPONENT_TYPE_CURVE};
  for (const GeometryComponentType type : types) {
    if (geometry_set.has(type)) {
      GeometryComponent &component = geometry_set.get_component_for_write(type);
      try_freeze_field_on_geometry(component, *id, domain, *field);
    }
  }

  set_output_field(params, *id, data_type);

  params.set_output("Geometry", geometry_set);
}

}  // namespace blender::nodes

void register_node_type_geo_attribute_freeze()
{
  static bNodeType ntype;

  geo_node_type_base(
      &ntype, GEO_NODE_ATTRIBUTE_FREEZE, "Attribute Freeze", NODE_CLASS_ATTRIBUTE, 0);
  node_type_socket_templates(&ntype, geo_node_attribute_freeze_in, geo_node_attribute_freeze_out);
  node_type_storage(&ntype,
                    "NodeGeometryAttributeFreeze",
                    node_free_standard_storage,
                    node_copy_standard_storage);
  node_type_init(&ntype, geo_node_attribute_freeze_init);
  node_type_update(&ntype, geo_node_attribute_freeze_update);
  ntype.geometry_node_execute = blender::nodes::geo_node_attribute_freeze_exec;
  ntype.draw_buttons = geo_node_attribute_freeze_layout;
  nodeRegisterType(&ntype);
}
