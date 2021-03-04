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

DerivedNodeTree::DerivedNodeTree(bNodeTree &btree, NodeTreeRefMap &node_tree_refs)
{
  root_context_ = &this->construct_context_recursively(nullptr, nullptr, btree, node_tree_refs);
}

DTreeContext &DerivedNodeTree::construct_context_recursively(DTreeContext *parent_context,
                                                             const NodeRef *parent_node,
                                                             bNodeTree &btree,
                                                             NodeTreeRefMap &node_tree_refs)
{
  DTreeContext &context = *allocator_.construct<DTreeContext>();
  context.parent_context_ = parent_context;
  context.parent_node_ = parent_node;
  context.tree_ = &get_tree_ref_from_map(node_tree_refs, btree);
  used_node_tree_refs_.add(context.tree_);

  for (const NodeRef *node : context.tree_->nodes()) {
    if (node->is_group_node()) {
      bNode *bnode = node->bnode();
      bNodeTree *child_btree = reinterpret_cast<bNodeTree *>(bnode->id);
      if (child_btree != nullptr) {
        DTreeContext &child = this->construct_context_recursively(
            &context, node, *child_btree, node_tree_refs);
        context.children_.add_new(node, &child);
      }
    }
  }

  return context;
}

DerivedNodeTree::~DerivedNodeTree()
{
  /* Has to be destructed manually, because the context info is allocated in a linear allocator. */
  this->destruct_context_recursively(root_context_);
}

void DerivedNodeTree::destruct_context_recursively(DTreeContext *context)
{
  for (DTreeContext *child : context->children_.values()) {
    this->destruct_context_recursively(child);
  }
  context->~DTreeContext();
}

bool DerivedNodeTree::has_link_cycles() const
{
  for (const NodeTreeRef *tree_ref : used_node_tree_refs_) {
    if (tree_ref->has_link_cycles()) {
      return true;
    }
  }
  return false;
}

void DerivedNodeTree::foreach_node(FunctionRef<void(DNode)> callback) const
{
  this->foreach_node_in_context_recursive(*root_context_, callback);
}

void DerivedNodeTree::foreach_node_in_context_recursive(const DTreeContext &context,
                                                        FunctionRef<void(DNode)> callback) const
{
  for (const NodeRef *node_ref : context.tree_->nodes()) {
    callback(DNode(&context, node_ref));
  }
  for (const DTreeContext *child_context : context.children_.values()) {
    this->foreach_node_in_context_recursive(*child_context, callback);
  }
}

DOutputSocket DInputSocket::get_corresponding_group_node_output() const
{
  BLI_assert(*this);
  BLI_assert(socket_ref_->node().is_group_output_node());
  BLI_assert(socket_ref_->index() < socket_ref_->node().inputs().size() - 1);

  const DTreeContext *parent_context = context_->parent_context();
  const NodeRef *parent_node = context_->parent_node();
  BLI_assert(parent_context != nullptr);
  BLI_assert(parent_node != nullptr);

  const int socket_index = socket_ref_->index();
  return {parent_context, &parent_node->output(socket_index)};
}

DOutputSocket DInputSocket::get_corresponding_group_input_socket() const
{
  BLI_assert(*this);
  BLI_assert(socket_ref_->node().is_group_node());

  const DTreeContext *child_context = context_->child_context(socket_ref_->node());
  BLI_assert(child_context != nullptr);

  const NodeTreeRef &child_tree = child_context->tree();
  Span<const NodeRef *> group_input_nodes = child_tree.nodes_by_type("NodeGroupInput");
  BLI_assert(!group_input_nodes.is_empty());

  const int socket_index = socket_ref_->index();
  return {child_context, &group_input_nodes[0]->output(socket_index)};
}

DInputSocket DOutputSocket::get_corresponding_group_node_input() const
{
  BLI_assert(*this);
  BLI_assert(socket_ref_->node().is_group_input_node());
  BLI_assert(socket_ref_->index() < socket_ref_->node().outputs().size() - 1);

  const DTreeContext *parent_context = context_->parent_context();
  const NodeRef *parent_node = context_->parent_node();
  BLI_assert(parent_context != nullptr);
  BLI_assert(parent_node != nullptr);

  const int socket_index = socket_ref_->index();
  return {parent_context, &parent_node->input(socket_index)};
}

DInputSocket DOutputSocket::get_corresponding_group_output_socket() const
{
  BLI_assert(*this);
  BLI_assert(socket_ref_->node().is_group_node());

  const DTreeContext *child_context = context_->child_context(socket_ref_->node());
  BLI_assert(child_context != nullptr);

  const NodeTreeRef &child_tree = child_context->tree();
  Span<const NodeRef *> group_output_nodes = child_tree.nodes_by_type("NodeGroupOutput");
  BLI_assert(!group_output_nodes.is_empty());

  const int socket_index = socket_ref_->index();
  return {child_context, &group_output_nodes[0]->input(socket_index)};
}

void DInputSocket::foreach_origin_socket(FunctionRef<void(DSocket)> callback) const
{
  BLI_assert(*this);
  for (const OutputSocketRef *linked_socket : socket_ref_->as_input().linked_sockets()) {
    const NodeRef &linked_node = linked_socket->node();
    DOutputSocket linked_dsocket{context_, linked_socket};

    if (linked_node.is_muted()) {
      for (const InternalLinkRef *internal_link : linked_node.internal_links()) {
        if (&internal_link->to() == linked_socket) {
          DInputSocket input_of_muted_node{context_, &internal_link->from()};
          input_of_muted_node.foreach_origin_socket(callback);
        }
      }
    }
    else if (linked_node.is_group_input_node()) {
      if (context_->is_root()) {
        callback(linked_dsocket);
      }
      else {
        DInputSocket socket_in_parent_group = linked_dsocket.get_corresponding_group_node_input();
        if (socket_in_parent_group->is_linked()) {
          socket_in_parent_group.foreach_origin_socket(callback);
        }
        else {
          callback(socket_in_parent_group);
        }
      }
    }
    else if (linked_node.is_group_node()) {
      DInputSocket socket_in_group = linked_dsocket.get_corresponding_group_output_socket();
      if (socket_in_group->is_linked()) {
        socket_in_group.foreach_origin_socket(callback);
      }
      else {
        callback(socket_in_group);
      }
    }
    else {
      callback(linked_dsocket);
    }
  }
}

void DOutputSocket::foreach_target_socket(FunctionRef<void(DInputSocket)> callback) const
{
  for (const InputSocketRef *linked_socket : socket_ref_->as_output().linked_sockets()) {
    const NodeRef &linked_node = linked_socket->node();
    DInputSocket linked_dsocket{context_, linked_socket};

    if (linked_node.is_muted()) {
      for (const InternalLinkRef *internal_link : linked_node.internal_links()) {
        if (&internal_link->from() == linked_socket) {
          DOutputSocket output_of_muted_node{context_, &internal_link->to()};
          output_of_muted_node.foreach_target_socket(callback);
        }
      }
    }
    else if (linked_node.is_group_output_node()) {
      if (context_->is_root()) {
        callback(linked_dsocket);
      }
      else {
        DOutputSocket socket_in_parent_group =
            linked_dsocket.get_corresponding_group_node_output();
        socket_in_parent_group.foreach_target_socket(callback);
      }
    }
    else if (linked_node.is_group_node()) {
      DOutputSocket socket_in_group = linked_dsocket.get_corresponding_group_input_socket();
      socket_in_group.foreach_target_socket(callback);
    }
    else {
      callback(linked_dsocket);
    }
  }
}

}  // namespace blender::nodes
