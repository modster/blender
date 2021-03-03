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

#include "NOD_XXX_node_tree.hh"

namespace blender::nodes {

XXXNodeTree::XXXNodeTree(bNodeTree &btree, NodeTreeRefMap &node_tree_refs)
{
  root_context_info_ = &this->construct_context_info_recursively(
      nullptr, nullptr, btree, node_tree_refs);
}

XXXNodeTreeContextInfo &XXXNodeTree::construct_context_info_recursively(
    XXXNodeTreeContextInfo *parent_context_info,
    const NodeRef *parent_node,
    bNodeTree &btree,
    NodeTreeRefMap &node_tree_refs)
{
  XXXNodeTreeContextInfo &context_info = *allocator_.construct<XXXNodeTreeContextInfo>();
  context_info.parent_context_info_ = parent_context_info;
  context_info.parent_node_ = parent_node;
  context_info.tree_ = &get_tree_ref_from_map(node_tree_refs, btree);

  for (const NodeRef *node : context_info.tree_->nodes()) {
    if (node->is_group_node()) {
      bNode *bnode = node->bnode();
      bNodeTree *child_btree = reinterpret_cast<bNodeTree *>(bnode->id);
      if (child_btree != nullptr) {
        XXXNodeTreeContextInfo &child = this->construct_context_info_recursively(
            &context_info, node, *child_btree, node_tree_refs);
        context_info.children_.add_new(node, &child);
      }
    }
  }

  return context_info;
}

XXXNodeTree::~XXXNodeTree()
{
  /* Has to be destructed manually, because the context info is allocated in a linear allocator. */
  this->destruct_context_info_recursively(root_context_info_);
}

void XXXNodeTree::destruct_context_info_recursively(XXXNodeTreeContextInfo *context_info)
{
  for (XXXNodeTreeContextInfo *child : context_info->children_.values()) {
    destruct_context_info_recursively(child);
  }
  context_info->~XXXNodeTreeContextInfo();
}

XXXOutputSocket XXXInputSocket::try_get_single_origin() const
{
  BLI_assert(*this);
  Span<const OutputSocketRef *> origins = socket->linked_sockets();
  if (origins.size() != 1) {
    return {};
  }
  const OutputSocketRef *origin = origins[0];
  return {context, origin};
}

XXXOutputSocket XXXInputSocket::get_corresponding_group_node_output() const
{
  BLI_assert(*this);
  BLI_assert(socket->node().is_group_output_node());
  BLI_assert(socket->index() < socket->node().inputs().size() - 1);

  const XXXNodeTreeContextInfo &context_info = context.info();
  const XXXNodeTreeContextInfo *parent_context_info = context_info.parent_context_info();
  const NodeRef *parent_node = context_info.parent_node();
  BLI_assert(parent_context_info != nullptr);
  BLI_assert(parent_node != nullptr);

  const int socket_index = socket->index();
  return {parent_context_info, &parent_node->output(socket_index)};
}

XXXOutputSocket XXXInputSocket::get_corresponding_group_input_socket() const
{
  BLI_assert(*this);
  BLI_assert(socket->node().is_group_node());

  const XXXNodeTreeContextInfo &context_info = context.info();
  const XXXNodeTreeContextInfo *child_context_info = context_info.child_context_info(
      socket->node());
  BLI_assert(child_context_info != nullptr);

  const NodeTreeRef &child_tree = child_context_info->tree();
  Span<const NodeRef *> group_input_nodes = child_tree.nodes_by_type("NodeGroupInput");
  BLI_assert(!group_input_nodes.is_empty());

  const int socket_index = socket->index();
  return {child_context_info, &group_input_nodes[0]->output(socket_index)};
}

XXXInputSocket XXXOutputSocket::get_corresponding_group_node_input() const
{
  BLI_assert(*this);
  BLI_assert(socket->node().is_group_input_node());
  BLI_assert(socket->index() < socket->node().outputs().size() - 1);

  const XXXNodeTreeContextInfo &context_info = context.info();
  const XXXNodeTreeContextInfo *parent_context_info = context_info.parent_context_info();
  const NodeRef *parent_node = context_info.parent_node();
  BLI_assert(parent_context_info != nullptr);
  BLI_assert(parent_node != nullptr);

  const int socket_index = socket->index();
  return {parent_context_info, &parent_node->input(socket_index)};
}

XXXInputSocket XXXOutputSocket::get_corresponding_group_output_socket() const
{
  BLI_assert(*this);
  BLI_assert(socket->node().is_group_node());

  const XXXNodeTreeContextInfo &context_info = context.info();
  const XXXNodeTreeContextInfo *child_context_info = context_info.child_context_info(
      socket->node());
  BLI_assert(child_context_info != nullptr);

  const NodeTreeRef &child_tree = child_context_info->tree();
  Span<const NodeRef *> group_output_nodes = child_tree.nodes_by_type("NodeGroupOutput");
  BLI_assert(!group_output_nodes.is_empty());

  const int socket_index = socket->index();
  return {child_context_info, &group_output_nodes[0]->input(socket_index)};
}

}  // namespace blender::nodes
