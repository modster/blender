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

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"

#include "BKE_material.h"

static bNodeSocketTemplate geo_node_geometry_expander_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

namespace blender::nodes {

static void geo_node_geometry_expander_layout(uiLayout *layout,
                                              bContext *UNUSED(C),
                                              PointerRNA *UNUSED(ptr))
{
  uiItemO(layout, "Add", ICON_ADD, "node.geometry_expander_output_add");
}

static void geo_node_geometry_expander_exec(GeoNodeExecParams params)
{
  const bNode &bnode = params.node();
  const NodeGeometryGeometryExpander *storage = (const NodeGeometryGeometryExpander *)
                                                    bnode.storage;
  LISTBASE_FOREACH (GeometryExpanderOutput *, expander_output, &storage->outputs) {
    params.set_output(expander_output->socket_identifier, 0.0f);
  }
}

static void geo_node_geometry_expander_init(bNodeTree *UNUSED(ntree), bNode *node)
{
  NodeGeometryGeometryExpander *storage = (NodeGeometryGeometryExpander *)MEM_callocN(
      sizeof(NodeGeometryGeometryExpander), __func__);
  node->storage = storage;
}

static void geo_node_geometry_expander_update(bNodeTree *ntree, bNode *node)
{
  NodeGeometryGeometryExpander *storage = (NodeGeometryGeometryExpander *)node->storage;

  Map<StringRef, bNodeSocket *> old_outputs;

  LISTBASE_FOREACH (bNodeSocket *, socket, &node->outputs) {
    old_outputs.add(socket->identifier, socket);
  }
  VectorSet<bNodeSocket *> new_sockets;
  LISTBASE_FOREACH (GeometryExpanderOutput *, expander_output, &storage->outputs) {
    bNodeSocket *socket = old_outputs.lookup_default(expander_output->socket_identifier, nullptr);
    if (socket == nullptr) {
      const char *idname = nodeStaticSocketType(expander_output->socket_type, PROP_NONE);
      socket = nodeAddSocket(ntree,
                             node,
                             SOCK_OUT,
                             idname,
                             expander_output->socket_identifier,
                             expander_output->data_identifier);
    }
    new_sockets.add_new(socket);
  }
  LISTBASE_FOREACH_MUTABLE (bNodeSocket *, socket, &node->outputs) {
    if (!new_sockets.contains(socket)) {
      nodeRemoveSocket(ntree, node, socket);
    }
  }
  BLI_listbase_clear(&node->outputs);
  for (bNodeSocket *socket : new_sockets) {
    BLI_addtail(&node->outputs, socket);
  }
}

static void geo_node_geometry_expander_storage_free(bNode *node)
{
  NodeGeometryGeometryExpander *storage = (NodeGeometryGeometryExpander *)node->storage;
  LISTBASE_FOREACH_MUTABLE (GeometryExpanderOutput *, expander_output, &storage->outputs) {
    MEM_freeN(expander_output->data_identifier);
    MEM_freeN(expander_output->socket_identifier);
    MEM_freeN(expander_output);
  }
  MEM_freeN(storage);
}

static void geo_node_geometry_expander_storage_copy(bNodeTree *UNUSED(dest_ntree),
                                                    bNode *dst_node,
                                                    const bNode *src_node)
{
  NodeGeometryGeometryExpander *src_storage = (NodeGeometryGeometryExpander *)src_node->storage;
  NodeGeometryGeometryExpander *dst_storage = (NodeGeometryGeometryExpander *)MEM_callocN(
      sizeof(NodeGeometryGeometryExpander), __func__);
  LISTBASE_FOREACH (GeometryExpanderOutput *, src_output, &src_storage->outputs) {
    GeometryExpanderOutput *dst_output = (GeometryExpanderOutput *)MEM_callocN(
        sizeof(GeometryExpanderOutput), __func__);
    *dst_output = *src_output;
    dst_output->data_identifier = (char *)MEM_dupallocN(src_output->data_identifier);
    dst_output->socket_identifier = (char *)MEM_dupallocN(src_output->socket_identifier);
    BLI_addtail(&dst_storage->outputs, dst_output);
  }
  dst_node->storage = dst_storage;
}

}  // namespace blender::nodes

void register_node_type_geo_geometry_expander()
{
  static bNodeType ntype;

  geo_node_type_base(
      &ntype, GEO_NODE_GEOMETRY_EXPANDER, "Geometry Expander", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, geo_node_geometry_expander_in, nullptr);
  node_type_init(&ntype, blender::nodes::geo_node_geometry_expander_init);
  node_type_storage(&ntype,
                    "NodeGeometryGeometryExpander",
                    blender::nodes::geo_node_geometry_expander_storage_free,
                    blender::nodes::geo_node_geometry_expander_storage_copy);
  node_type_update(&ntype, blender::nodes::geo_node_geometry_expander_update);
  ntype.geometry_node_execute = blender::nodes::geo_node_geometry_expander_exec;
  ntype.draw_buttons = blender::nodes::geo_node_geometry_expander_layout;
  nodeRegisterType(&ntype);
}
