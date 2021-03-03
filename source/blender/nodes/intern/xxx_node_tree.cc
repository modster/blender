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
  root_context_ = &this->construct_context_recursively(nullptr, nullptr, btree, node_tree_refs);
}

XXXNodeTreeContext &XXXNodeTree::construct_context_recursively(XXXNodeTreeContext *parent_context,
                                                               const NodeRef *parent_node,
                                                               bNodeTree &btree,
                                                               NodeTreeRefMap &node_tree_refs)
{
  XXXNodeTreeContext &context = *allocator_.construct<XXXNodeTreeContext>();
  context.parent_context_ = parent_context;
  context.parent_node_ = parent_node;
  context.tree_ = &get_tree_ref_from_map(node_tree_refs, btree);

  for (const NodeRef *node : context.tree_->nodes()) {
    if (node->is_group_node()) {
      bNode *bnode = node->bnode();
      bNodeTree *child_btree = reinterpret_cast<bNodeTree *>(bnode->id);
      if (child_btree != nullptr) {
        XXXNodeTreeContext &child = this->construct_context_recursively(
            &context, node, *child_btree, node_tree_refs);
        context.children_.add_new(node, &child);
      }
    }
  }

  return context;
}

XXXNodeTree::~XXXNodeTree()
{
  /* Has to be destructed manually, because the context info is allocated in a linear allocator. */
  this->destruct_context_recursively(root_context_);
}

void XXXNodeTree::destruct_context_recursively(XXXNodeTreeContext *context)
{
  for (XXXNodeTreeContext *child : context->children_.values()) {
    this->destruct_context_recursively(child);
  }
  context->~XXXNodeTreeContext();
}

XXXOutputSocket XXXInputSocket::get_corresponding_group_node_output() const
{
  BLI_assert(*this);
  BLI_assert(socket->node().is_group_output_node());
  BLI_assert(socket->index() < socket->node().inputs().size() - 1);

  const XXXNodeTreeContext *parent_context = context->parent_context();
  const NodeRef *parent_node = context->parent_node();
  BLI_assert(parent_context != nullptr);
  BLI_assert(parent_node != nullptr);

  const int socket_index = socket->index();
  return {parent_context, &parent_node->output(socket_index)};
}

XXXOutputSocket XXXInputSocket::get_corresponding_group_input_socket() const
{
  BLI_assert(*this);
  BLI_assert(socket->node().is_group_node());

  const XXXNodeTreeContext *child_context = context->child_context(socket->node());
  BLI_assert(child_context != nullptr);

  const NodeTreeRef &child_tree = child_context->tree();
  Span<const NodeRef *> group_input_nodes = child_tree.nodes_by_type("NodeGroupInput");
  BLI_assert(!group_input_nodes.is_empty());

  const int socket_index = socket->index();
  return {child_context, &group_input_nodes[0]->output(socket_index)};
}

XXXInputSocket XXXOutputSocket::get_corresponding_group_node_input() const
{
  BLI_assert(*this);
  BLI_assert(socket->node().is_group_input_node());
  BLI_assert(socket->index() < socket->node().outputs().size() - 1);

  const XXXNodeTreeContext *parent_context = context->parent_context();
  const NodeRef *parent_node = context->parent_node();
  BLI_assert(parent_context != nullptr);
  BLI_assert(parent_node != nullptr);

  const int socket_index = socket->index();
  return {parent_context, &parent_node->input(socket_index)};
}

XXXInputSocket XXXOutputSocket::get_corresponding_group_output_socket() const
{
  BLI_assert(*this);
  BLI_assert(socket->node().is_group_node());

  const XXXNodeTreeContext *child_context = context->child_context(socket->node());
  BLI_assert(child_context != nullptr);

  const NodeTreeRef &child_tree = child_context->tree();
  Span<const NodeRef *> group_output_nodes = child_tree.nodes_by_type("NodeGroupOutput");
  BLI_assert(!group_output_nodes.is_empty());

  const int socket_index = socket->index();
  return {child_context, &group_output_nodes[0]->input(socket_index)};
}

}  // namespace blender::nodes
