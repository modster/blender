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

static bNodeSocketTemplate geo_node_attribute_extract_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_STRING, N_("Attribute")},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_attribute_extract_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_VECTOR, N_("Value")},
    {SOCK_FLOAT, N_("Value")},
    {SOCK_RGBA, N_("Value")},
    {SOCK_BOOLEAN, N_("Value")},
    {SOCK_INT, N_("Value")},
    {-1, ""},
};

static void geo_node_attribute_extract_layout(uiLayout *layout,
                                              bContext *UNUSED(C),
                                              PointerRNA *ptr)
{
  uiItemR(layout, ptr, "data_type", 0, "", ICON_NONE);
  uiItemR(layout, ptr, "delete_persistent", 0, nullptr, ICON_NONE);
}

static void geo_node_attribute_extract_init(bNodeTree *UNUSED(tree), bNode *node)
{
  NodeGeometryAttributeExtract *storage = (NodeGeometryAttributeExtract *)MEM_callocN(
      sizeof(NodeGeometryAttributeExtract), __func__);
  storage->data_type = CD_PROP_FLOAT;
  storage->delete_persistent = false;
  node->storage = storage;
}

static void geo_node_attribute_extract_update(bNodeTree *UNUSED(ntree), bNode *node)
{

  const NodeGeometryAttributeExtract &storage = *(const NodeGeometryAttributeExtract *)
                                                     node->storage;

  bNodeSocket *socket_value_vector = (bNodeSocket *)BLI_findlink(&node->outputs, 1);
  bNodeSocket *socket_value_float = socket_value_vector->next;
  bNodeSocket *socket_value_color4f = socket_value_float->next;
  bNodeSocket *socket_value_boolean = socket_value_color4f->next;
  bNodeSocket *socket_value_int32 = socket_value_boolean->next;

  const CustomDataType data_type = (CustomDataType)storage.data_type;

  nodeSetSocketAvailability(socket_value_vector, data_type == CD_PROP_FLOAT3);
  nodeSetSocketAvailability(socket_value_float, data_type == CD_PROP_FLOAT);
  nodeSetSocketAvailability(socket_value_color4f, data_type == CD_PROP_COLOR);
  nodeSetSocketAvailability(socket_value_boolean, data_type == CD_PROP_BOOL);
  nodeSetSocketAvailability(socket_value_int32, data_type == CD_PROP_INT32);
}

namespace blender::nodes {

static void convert_attribute(GeometryComponent &component,
                              const StringRef attribute_name,
                              const AnonymousCustomDataLayerID &layer_id,
                              bool delete_persistent)
{
  ReadAttributeLookup attribute_lookup = component.attribute_try_get_for_read(attribute_name);
  if (!attribute_lookup) {
    return;
  }
  const GVArray &varray = *attribute_lookup.varray;
  const CPPType &cpp_type = varray.type();
  const CustomDataType data_type = bke::cpp_type_to_custom_data_type(cpp_type);
  component.attribute_try_create_anonymous(
      layer_id, attribute_lookup.domain, data_type, AttributeInitVArray(&varray));

  if (delete_persistent) {
    component.attribute_try_delete(attribute_name);
  }
}

static void geo_node_attribute_extract_exec(GeoNodeExecParams params)
{
  const bNode &node = params.node();
  const NodeGeometryAttributeExtract &storage = *(const NodeGeometryAttributeExtract *)
                                                     node.storage;
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");

  geometry_set = geometry_set_realize_instances(geometry_set);

  const CustomDataType data_type = static_cast<CustomDataType>(storage.data_type);
  const CPPType *cpp_type = bke::custom_data_type_to_cpp_type(data_type);
  const bool delete_persistent = storage.delete_persistent;

  const std::string attribute_name = params.get_input<std::string>("Attribute");
  AnonymousCustomDataLayerID *layer_id = CustomData_anonymous_id_new(attribute_name.c_str());
  auto output_field = new bke::AnonymousAttributeField(*layer_id, *cpp_type);

  if (geometry_set.has<MeshComponent>()) {
    convert_attribute(geometry_set.get_component_for_write<MeshComponent>(),
                      attribute_name,
                      *layer_id,
                      delete_persistent);
  }
  if (geometry_set.has<PointCloudComponent>()) {
    convert_attribute(geometry_set.get_component_for_write<PointCloudComponent>(),
                      attribute_name,
                      *layer_id,
                      delete_persistent);
  }
  if (geometry_set.has<CurveComponent>()) {
    convert_attribute(geometry_set.get_component_for_write<CurveComponent>(),
                      attribute_name,
                      *layer_id,
                      delete_persistent);
  }

  params.set_output("Geometry", geometry_set);

  switch (data_type) {
    case CD_PROP_FLOAT: {
      params.set_output("Value_001", bke::FieldRef<float>(output_field));
      break;
    }
    case CD_PROP_FLOAT3: {
      params.set_output("Value", bke::FieldRef<float3>(output_field));
      break;
    }
    case CD_PROP_INT32: {
      params.set_output("Value_004", bke::FieldRef<int>(output_field));
      break;
    }
    case CD_PROP_BOOL: {
      params.set_output("Value_003", bke::FieldRef<bool>(output_field));
      break;
    }
    case CD_PROP_COLOR: {
      params.set_output("Value_002", bke::FieldRef<ColorGeometry4f>(output_field));
      break;
    }
    default: {
      BLI_assert_unreachable();
    }
  }
}

}  // namespace blender::nodes

void register_node_type_geo_attribute_extract()
{
  static bNodeType ntype;

  geo_node_type_base(
      &ntype, GEO_NODE_ATTRIBUTE_EXTRACT, "Attribute Extract", NODE_CLASS_ATTRIBUTE, 0);
  node_type_socket_templates(
      &ntype, geo_node_attribute_extract_in, geo_node_attribute_extract_out);
  node_type_init(&ntype, geo_node_attribute_extract_init);
  node_type_update(&ntype, geo_node_attribute_extract_update);
  node_type_storage(&ntype,
                    "NodeGeometryAttributeExtract",
                    node_free_standard_storage,
                    node_copy_standard_storage);
  ntype.geometry_node_execute = blender::nodes::geo_node_attribute_extract_exec;
  ntype.draw_buttons = geo_node_attribute_extract_layout;
  nodeRegisterType(&ntype);
}
