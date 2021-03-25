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

#include "node_geometry_util.hh"

#include "BLI_heap_value.hh"
#include "BLI_rand.hh"

#include "DNA_mesh_types.h"
#include "DNA_pointcloud_types.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "WM_types.h"

static bNodeSocketTemplate geo_node_attribute_fill_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_STRING, N_("Attribute")},
    {SOCK_VECTOR, N_("Value"), 0.0f, 0.0f, 0.0f, 0.0f, -FLT_MAX, FLT_MAX},
    {SOCK_FLOAT, N_("Value"), 0.0f, 0.0f, 0.0f, 0.0f, -FLT_MAX, FLT_MAX},
    {SOCK_RGBA, N_("Value"), 0.0f, 0.0f, 0.0f, 0.0f, -FLT_MAX, FLT_MAX},
    {SOCK_BOOLEAN, N_("Value"), 0.0f, 0.0f, 0.0f, 0.0f, -FLT_MAX, FLT_MAX},
    {SOCK_INT, N_("Value"), 0, 0, 0, 0, -10000000.0f, 10000000.0f},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_attribute_fill_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

static void geo_node_attribute_fill_init(bNodeTree *UNUSED(tree), bNode *node)
{
  node->custom1 = CD_PROP_FLOAT;
  node->custom2 = ATTR_DOMAIN_AUTO;

  LISTBASE_FOREACH (bNodeSocket *, socket, &node->inputs) {
    if (socket->type != SOCK_GEOMETRY) {
      socket->flag |= SOCK_HIDDEN;
    }
  }
}

static void geo_node_attribute_fill_update(bNodeTree *UNUSED(ntree), bNode *node)
{
  bNodeSocket *socket_value_vector = (bNodeSocket *)BLI_findlink(&node->inputs, 2);
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

struct SocketMenuInfo {
  bNodeTree *ntree;
  bNode *node;
  bNodeSocket *socket;
  std::string enum_name;

  uint64_t hash() const
  {
    return default_hash_3(ntree, node, socket);
  }

  friend bool operator==(const SocketMenuInfo &a, const SocketMenuInfo &b)
  {
    return a.ntree == b.ntree && a.node == b.node && a.socket == b.socket;
  }
};

static Set<HeapValue<SocketMenuInfo>> &get_socket_menu_info_set()
{
  static Set<HeapValue<SocketMenuInfo>> set;
  return set;
}

static void draw_socket_menu(bContext *UNUSED(C), uiLayout *layout, void *arg)
{
  SocketMenuInfo *socket_info = (SocketMenuInfo *)arg;

  PointerRNA node_ptr;
  RNA_pointer_create(&socket_info->ntree->id, &RNA_Node, socket_info->node, &node_ptr);
  PointerRNA socket_ptr;
  RNA_pointer_create(&socket_info->ntree->id, &RNA_NodeSocket, socket_info->socket, &socket_ptr);

  if (socket_info->socket->flag & SOCK_HIDDEN) {
    PointerRNA expose_props;
    uiItemFullO(layout,
                "node.expose_input_socket",
                "Expose",
                ICON_TRACKING_BACKWARDS_SINGLE,
                nullptr,
                WM_OP_EXEC_DEFAULT,
                0,
                &expose_props);
    RNA_string_set(&expose_props, "tree_name", socket_info->ntree->id.name + 2);
    RNA_string_set(&expose_props, "node_name", socket_info->node->name);
    RNA_string_set(&expose_props, "socket_name", socket_info->socket->name);
    RNA_boolean_set(&expose_props, "expose", true);
  }
  else {
    uiLayout *col = uiLayoutColumn(layout, false);
    uiLayoutSetEnabled(col, (socket_info->socket->flag & SOCK_IN_USE) == 0);
    PointerRNA expose_props;
    uiItemFullO(col,
                "node.expose_input_socket",
                "Unexpose",
                ICON_TRACKING_CLEAR_BACKWARDS,
                nullptr,
                WM_OP_EXEC_DEFAULT,
                0,
                &expose_props);
    RNA_string_set(&expose_props, "tree_name", socket_info->ntree->id.name + 2);
    RNA_string_set(&expose_props, "node_name", socket_info->node->name);
    RNA_string_set(&expose_props, "socket_name", socket_info->socket->name);
    RNA_boolean_set(&expose_props, "expose", false);
  }

  if (!socket_info->enum_name.empty()) {
    uiItemsEnumR(layout, &node_ptr, socket_info->enum_name.c_str());
  }
}

static void draw_input_socket(bContext *C,
                              uiLayout *layout,
                              PointerRNA *node_ptr,
                              StringRef socket_name,
                              StringRef additional_enum_prop = "")
{
  bNodeTree *ntree = (bNodeTree *)node_ptr->owner_id;
  bNode *node = (bNode *)node_ptr->data;

  bNodeSocket *socket_to_draw = nullptr;
  LISTBASE_FOREACH (bNodeSocket *, socket, &node->inputs) {
    if ((socket->flag & SOCK_UNAVAIL) == 0 && socket->name == socket_name) {
      socket_to_draw = socket;
      break;
    }
  }
  if (socket_to_draw == nullptr) {
    return;
  }
  PointerRNA socket_ptr;
  RNA_pointer_create(&ntree->id, &RNA_NodeSocket, socket_to_draw, &socket_ptr);

  SocketMenuInfo info;
  info.ntree = ntree;
  info.node = node;
  info.socket = socket_to_draw;
  info.enum_name = additional_enum_prop;

  Set<HeapValue<SocketMenuInfo>> &set = get_socket_menu_info_set();
  const SocketMenuInfo *stored_info = set.lookup_key_or_add_as(info).get();
  uiLayout *row = uiLayoutRow(layout, false);
  uiLayout *sub_row = uiLayoutRow(row, false);
  uiLayoutSetActive(sub_row, (socket_to_draw->flag & SOCK_HIDDEN) != 0);
  socket_to_draw->typeinfo->draw(C, sub_row, &socket_ptr, node_ptr, socket_to_draw->name);
  uiItemMenuF(row, "", ICON_DOWNARROW_HLT, draw_socket_menu, (void *)stored_info);
}

static void geo_node_attribute_fill_layout(uiLayout *layout, bContext *C, PointerRNA *node_ptr)
{
  uiItemR(layout, node_ptr, "domain", 0, IFACE_("Domain"), ICON_NONE);
  draw_input_socket(C, layout, node_ptr, "Attribute");
  draw_input_socket(C, layout, node_ptr, "Value", "data_type");
}

static AttributeDomain get_result_domain(const GeometryComponent &component,
                                         StringRef attribute_name)
{
  /* Use the domain of the result attribute if it already exists. */
  ReadAttributePtr result_attribute = component.attribute_try_get_for_read(attribute_name);
  if (result_attribute) {
    return result_attribute->domain();
  }
  return ATTR_DOMAIN_POINT;
}

static void fill_attribute(GeometryComponent &component, const GeoNodeExecParams &params)
{
  const std::string attribute_name = params.get_input<std::string>("Attribute");
  if (attribute_name.empty()) {
    return;
  }

  const bNode &node = params.node();
  const CustomDataType data_type = static_cast<CustomDataType>(node.custom1);
  const AttributeDomain domain = static_cast<AttributeDomain>(node.custom2);
  const AttributeDomain result_domain = (domain == ATTR_DOMAIN_AUTO) ?
                                            get_result_domain(component, attribute_name) :
                                            domain;

  OutputAttributePtr attribute = component.attribute_try_get_for_output(
      attribute_name, result_domain, data_type);
  if (!attribute) {
    return;
  }

  switch (data_type) {
    case CD_PROP_FLOAT: {
      const float value = params.get_input<float>("Value_001");
      MutableSpan<float> attribute_span = attribute->get_span_for_write_only<float>();
      attribute_span.fill(value);
      break;
    }
    case CD_PROP_FLOAT3: {
      const float3 value = params.get_input<float3>("Value");
      MutableSpan<float3> attribute_span = attribute->get_span_for_write_only<float3>();
      attribute_span.fill(value);
      break;
    }
    case CD_PROP_COLOR: {
      const Color4f value = params.get_input<Color4f>("Value_002");
      MutableSpan<Color4f> attribute_span = attribute->get_span_for_write_only<Color4f>();
      attribute_span.fill(value);
      break;
    }
    case CD_PROP_BOOL: {
      const bool value = params.get_input<bool>("Value_003");
      MutableSpan<bool> attribute_span = attribute->get_span_for_write_only<bool>();
      attribute_span.fill(value);
      break;
    }
    case CD_PROP_INT32: {
      const int value = params.get_input<int>("Value_004");
      MutableSpan<int> attribute_span = attribute->get_span_for_write_only<int>();
      attribute_span.fill(value);
    }
    default:
      break;
  }

  attribute.apply_span_and_save();
}

static void geo_node_attribute_fill_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");

  geometry_set = geometry_set_realize_instances(geometry_set);

  if (geometry_set.has<MeshComponent>()) {
    fill_attribute(geometry_set.get_component_for_write<MeshComponent>(), params);
  }
  if (geometry_set.has<PointCloudComponent>()) {
    fill_attribute(geometry_set.get_component_for_write<PointCloudComponent>(), params);
  }

  params.set_output("Geometry", geometry_set);
}

}  // namespace blender::nodes

void register_node_type_geo_attribute_fill()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_ATTRIBUTE_FILL, "Attribute Fill", NODE_CLASS_ATTRIBUTE, 0);
  node_type_socket_templates(&ntype, geo_node_attribute_fill_in, geo_node_attribute_fill_out);
  node_type_init(&ntype, geo_node_attribute_fill_init);
  node_type_update(&ntype, geo_node_attribute_fill_update);
  ntype.geometry_node_execute = blender::nodes::geo_node_attribute_fill_exec;
  ntype.draw_buttons_ex = blender::nodes::geo_node_attribute_fill_layout;
  nodeRegisterType(&ntype);
}
