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

#include "UI_interface.h"
#include "UI_resources.h"

#include "WM_types.h"

#include "RNA_access.h"

static bNodeSocketTemplate node_portal_sockets[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_FLOAT, N_("Float"), 0.0f, 0.0f, 0.0f, 0.0f, -FLT_MAX, FLT_MAX},
    {SOCK_VECTOR, N_("Vector"), 0.0f, 0.0f, 0.0f, 0.0f, -FLT_MAX, FLT_MAX},
    {SOCK_RGBA, N_("Color"), 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f},
    {SOCK_INT, N_("Integer"), 0, 0, 0, 0, -10000, 10000},
    {SOCK_BOOLEAN, N_("Boolean")},
    {SOCK_STRING, N_("String")},
    {SOCK_OBJECT, N_("Object")},
    {SOCK_COLLECTION, N_("Collection")},
    {-1, ""},
};

// static bool output_with_id_exists(const bNodeTree &ntree, const int portal_id)
// {
//   LISTBASE_FOREACH (bNode *, node, &ntree.nodes) {
//     if (node->type == NODE_PORTAL_OUT) {
//       if (((NodePortalOut *)node->storage)->portal_id == portal_id) {
//         return true;
//       }
//     }
//   }
//   return false;
// }

static void node_portal_in_layout(uiLayout *UNUSED(layout),
                                  bContext *UNUSED(C),
                                  PointerRNA *UNUSED(ptr))
{
  // uiItemR(layout, ptr, "portal_id", 0, IFACE_("Portal ID"), ICON_NONE);

  // bNodeTree &ntree = *(bNodeTree *)ptr->owner_id;
  // bNode *node = (bNode *)ptr->data;
  // int portal_id = RNA_int_get(ptr, "portal_id");

  // if (!output_with_id_exists(ntree, portal_id)) {
  //   PointerRNA op_ptr;
  //   uiItemFullO(layout,
  //               "node.add_node",
  //               "Add Endpoint",
  //               ICON_NONE,
  //               nullptr,
  //               WM_OP_INVOKE_DEFAULT,
  //               0,
  //               &op_ptr);
  //   RNA_string_set(&op_ptr, "type", "NodePortalOut");
  //   RNA_boolean_set(&op_ptr, "use_transform", true);
  //   PropertyRNA *settings_prop = RNA_struct_find_property(&op_ptr, "settings");
  //   PointerRNA item_ptr;
  //   RNA_property_collection_add(&op_ptr, settings_prop, &item_ptr);
  //   RNA_string_set(&item_ptr, "name", "portal_id");
  //   std::string portal_id_str = std::to_string(RNA_int_get(ptr, "portal_id"));
  //   RNA_string_set(&item_ptr, "value", portal_id_str.c_str());
  // }
}

static void node_portal_out_layout(uiLayout *UNUSED(layout),
                                   bContext *UNUSED(C),
                                   PointerRNA *UNUSED(ptr))
{
  // uiItemR(layout, ptr, "portal_id", 0, IFACE_("Portal ID"), ICON_NONE);
}

namespace blender::nodes {

static void node_portal_in_init(bNodeTree *UNUSED(tree), bNode *node)
{
  NodePortalIn *data = (NodePortalIn *)MEM_callocN(sizeof(NodePortalIn), __func__);
  data->portal_id = rand();
  node->storage = data;
}

static void node_portal_out_init(bNodeTree *UNUSED(tree), bNode *node)
{
  NodePortalOut *data = (NodePortalOut *)MEM_callocN(sizeof(NodePortalOut), __func__);
  node->storage = data;
}

static void node_portal_in_update(bNodeTree *UNUSED(ntree), bNode *node)
{
  NodePortalIn &node_storage = *(NodePortalIn *)node->storage;
  UNUSED_VARS(node_storage);
}

static void node_portal_out_update(bNodeTree *UNUSED(ntree), bNode *node)
{
  NodePortalOut &node_storage = *(NodePortalOut *)node->storage;
  UNUSED_VARS(node_storage);
}

}  // namespace blender::nodes

void register_node_type_portal_in()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, NODE_PORTAL_IN, "Portal In", NODE_CLASS_LAYOUT, 0);
  node_type_socket_templates(&ntype, node_portal_sockets, nullptr);
  node_type_init(&ntype, blender::nodes::node_portal_in_init);
  node_type_update(&ntype, blender::nodes::node_portal_in_update);
  node_type_storage(
      &ntype, "NodePortalIn", node_free_standard_storage, node_copy_standard_storage);
  ntype.draw_buttons = node_portal_in_layout;
  nodeRegisterType(&ntype);
}

void register_node_type_portal_out()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, NODE_PORTAL_OUT, "Portal Out", NODE_CLASS_LAYOUT, 0);
  node_type_socket_templates(&ntype, nullptr, node_portal_sockets);
  node_type_init(&ntype, blender::nodes::node_portal_out_init);
  node_type_update(&ntype, blender::nodes::node_portal_out_update);
  node_type_storage(
      &ntype, "NodePortalOut", node_free_standard_storage, node_copy_standard_storage);
  ntype.draw_buttons = node_portal_out_layout;
  nodeRegisterType(&ntype);
}
