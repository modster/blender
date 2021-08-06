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

static bNodeSocketTemplate geo_node_attribute_store_anonymous_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_VECTOR,
     N_("Value"),
     0.0f,
     0.0f,
     0.0f,
     0.0f,
     -FLT_MAX,
     FLT_MAX,
     PROP_NONE,
     SOCK_IS_FIELD},
    {SOCK_FLOAT, N_("Value"), 0.0f, 0.0f, 0.0f, 0.0f, -FLT_MAX, FLT_MAX, PROP_NONE, SOCK_IS_FIELD},
    {SOCK_RGBA, N_("Value"), 0.0f, 0.0f, 0.0f, 0.0f, -FLT_MAX, FLT_MAX, PROP_NONE, SOCK_IS_FIELD},
    {SOCK_BOOLEAN,
     N_("Value"),
     0.0f,
     0.0f,
     0.0f,
     0.0f,
     -FLT_MAX,
     FLT_MAX,
     PROP_NONE,
     SOCK_IS_FIELD},
    {SOCK_INT, N_("Value"), 0, 0, 0, 0, -10000000.0f, 10000000.0f, PROP_NONE, SOCK_IS_FIELD},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_attribute_store_anonymous_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_VECTOR, N_("Attribute"), 0.0f, 0.0f, 0.0f, 0.0f, -FLT_MAX, FLT_MAX},
    {SOCK_FLOAT, N_("Attribute"), 0.0f, 0.0f, 0.0f, 0.0f, -FLT_MAX, FLT_MAX},
    {SOCK_RGBA, N_("Attribute"), 0.0f, 0.0f, 0.0f, 0.0f, -FLT_MAX, FLT_MAX},
    {SOCK_BOOLEAN, N_("Attribute"), 0.0f, 0.0f, 0.0f, 0.0f, -FLT_MAX, FLT_MAX},
    {SOCK_INT, N_("Attribute"), 0, 0, 0, 0, -10000000.0f, 10000000.0f},
    {-1, ""},
};

static void geo_node_attribute_store_anonymous_layout(uiLayout *layout,
                                                      bContext *UNUSED(C),
                                                      PointerRNA *ptr)
{
  uiLayoutSetPropSep(layout, true);
  uiLayoutSetPropDecorate(layout, false);
  uiItemR(layout, ptr, "domain", 0, "", ICON_NONE);
  uiItemR(layout, ptr, "data_type", 0, "", ICON_NONE);
}

static void geo_node_attribute_store_anonymous_init(bNodeTree *UNUSED(tree), bNode *node)
{
  NodeGeometryAttributeStore *data = (NodeGeometryAttributeStore *)MEM_callocN(
      sizeof(NodeGeometryAttributeStore), __func__);
  data->data_type = CD_PROP_FLOAT;
  data->domain = ATTR_DOMAIN_POINT;

  node->storage = data;
}

static void geo_node_attribute_store_anonymous_update(bNodeTree *UNUSED(ntree), bNode *node)
{
  const NodeGeometryAttributeStore &storage = *(const NodeGeometryAttributeStore *)node->storage;
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

template<typename T>
void fill_attribute_impl(GeometryComponent &component,
                         OutputAttribute &attribute,
                         const GeoNodeExecParams &params,
                         const StringRef input_name)
{
  const AttributeDomain domain = attribute.domain();
  const int domain_size = attribute->size();
  bke::FieldRef<T> value_field = params.get_input_field<T>(input_name);
  bke::FieldInputs field_inputs = value_field->prepare_inputs();
  Vector<std::unique_ptr<bke::FieldInputValue>> input_values;
  prepare_field_inputs(field_inputs, component, domain, input_values);
  bke::FieldOutput field_output = value_field->evaluate(IndexMask(domain_size), field_inputs);
  for (const int i : IndexRange(domain_size)) {
    T value;
    field_output.varray_ref().get(i, &value);
    attribute->set_by_copy(i, &value);
  }
}

static void fill_attribute_data(GeometryComponent &component,
                                const GeoNodeExecParams &params,
                                OutputAttribute &attribute,
                                const CustomDataType data_type)
{
  switch (data_type) {
    case CD_PROP_FLOAT: {
      fill_attribute_impl<float>(component, attribute, params, "Value_001");
      break;
    }
    case CD_PROP_FLOAT3: {
      fill_attribute_impl<float3>(component, attribute, params, "Value");
      break;
    }
    case CD_PROP_COLOR: {
      fill_attribute_impl<ColorGeometry4f>(component, attribute, params, "Value_002");
      break;
    }
    case CD_PROP_BOOL: {
      fill_attribute_impl<bool>(component, attribute, params, "Value_003");
      break;
    }
    case CD_PROP_INT32: {
      fill_attribute_impl<int>(component, attribute, params, "Value_004");
      break;
    }
    default:
      break;
  }
}

static void fill_anonymous(GeometryComponent &component,
                           const GeoNodeExecParams &params,
                           const AnonymousCustomDataLayerID &layer_id,
                           const AttributeDomain domain,
                           const CustomDataType data_type)
{
  OutputAttribute attribute = component.attribute_try_get_anonymous_for_output(
      layer_id, domain, data_type);
  if (!attribute) {
    return;
  }

  fill_attribute_data(component, params, attribute, data_type);

  attribute.save();
}

static void geo_node_attribute_store_anonymous_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");

  geometry_set = bke::geometry_set_realize_instances(geometry_set);

  const bNode &node = params.node();
  const NodeGeometryAttributeStore &storage = *(const NodeGeometryAttributeStore *)node.storage;
  const CustomDataType data_type = static_cast<CustomDataType>(storage.data_type);
  const AttributeDomain domain = static_cast<AttributeDomain>(storage.domain);

  AnonymousCustomDataLayerID *id = CustomData_anonymous_id_new("Store Attribute");

  static const Array<GeometryComponentType> types = {
      GEO_COMPONENT_TYPE_MESH, GEO_COMPONENT_TYPE_POINT_CLOUD, GEO_COMPONENT_TYPE_CURVE};
  for (const GeometryComponentType type : types) {
    if (geometry_set.has(type)) {
      fill_anonymous(geometry_set.get_component_for_write(type), params, *id, domain, data_type);
    }
  }

  set_output_field(params, *id, data_type);

  params.set_output("Geometry", geometry_set);
}

}  // namespace blender::nodes

void register_node_type_geo_attribute_store_anonymous()
{
  static bNodeType ntype;

  geo_node_type_base(
      &ntype, GEO_NODE_ATTRIBUTE_STORE_ANONYMOUS, "Store Attribute", NODE_CLASS_ATTRIBUTE, 0);
  node_type_socket_templates(
      &ntype, geo_node_attribute_store_anonymous_in, geo_node_attribute_store_anonymous_out);
  node_type_storage(&ntype,
                    "NodeGeometryAttributeStore",
                    node_free_standard_storage,
                    node_copy_standard_storage);
  node_type_init(&ntype, geo_node_attribute_store_anonymous_init);
  node_type_update(&ntype, geo_node_attribute_store_anonymous_update);
  ntype.geometry_node_execute = blender::nodes::geo_node_attribute_store_anonymous_exec;
  ntype.draw_buttons = geo_node_attribute_store_anonymous_layout;
  nodeRegisterType(&ntype);
}
