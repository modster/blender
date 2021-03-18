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

static bNodeSocketTemplate node_portal_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

static bNodeSocketTemplate node_portal_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

static void node_portal_in_layout(uiLayout *UNUSED(layout),
                                  bContext *UNUSED(C),
                                  PointerRNA *UNUSED(ptr))
{
}

static void node_portal_out_layout(uiLayout *UNUSED(layout),
                                   bContext *UNUSED(C),
                                   PointerRNA *UNUSED(ptr))
{
}

namespace blender::nodes {

static void node_portal_in_init(bNodeTree *UNUSED(tree), bNode *node)
{
  NodePortalIn *data = (NodePortalIn *)MEM_callocN(sizeof(NodePortalIn), __func__);
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

  geo_node_type_base(&ntype, NODE_PORTAL_IN, "Portal In", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, node_portal_in, nullptr);
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

  geo_node_type_base(&ntype, NODE_PORTAL_OUT, "Portal Out", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, nullptr, node_portal_out);
  node_type_init(&ntype, blender::nodes::node_portal_out_init);
  node_type_update(&ntype, blender::nodes::node_portal_out_update);
  node_type_storage(
      &ntype, "NodePortalOut", node_free_standard_storage, node_copy_standard_storage);
  ntype.draw_buttons = node_portal_out_layout;
  nodeRegisterType(&ntype);
}
