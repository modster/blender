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

#pragma once

#include "BLI_function_ref.hh"
#include "BLI_vector_set.hh"

#include "NOD_node_tree_ref.hh"

namespace blender::nodes {

class XXXNodeTreeContext;
class XXXNodeTree;

struct XXXNode;
struct XXXSocket;
struct XXXInputSocket;
struct XXXOutputSocket;

class XXXNodeTreeContext {
 private:
  XXXNodeTreeContext *parent_context_;
  const NodeRef *parent_node_;
  const NodeTreeRef *tree_;
  Map<const NodeRef *, XXXNodeTreeContext *> children_;

  friend XXXNodeTree;

 public:
  const NodeTreeRef &tree() const;
  const XXXNodeTreeContext *parent_context() const;
  const NodeRef *parent_node() const;
  const XXXNodeTreeContext *child_context(const NodeRef &node) const;
  bool is_root() const;
};

class XXXNode {
 private:
  const XXXNodeTreeContext *context_ = nullptr;
  const NodeRef *node_ref_ = nullptr;

 public:
  XXXNode() = default;
  XXXNode(const XXXNodeTreeContext *context, const NodeRef *node);

  const XXXNodeTreeContext *context() const;
  const NodeRef *node_ref() const;
  const NodeRef *operator->() const;

  friend bool operator==(const XXXNode &a, const XXXNode &b);
  friend bool operator!=(const XXXNode &a, const XXXNode &b);
  operator bool() const;

  uint64_t hash() const;
};

class XXXSocket {
 protected:
  const XXXNodeTreeContext *context_ = nullptr;
  const SocketRef *socket_ref_ = nullptr;

 public:
  XXXSocket() = default;
  XXXSocket(const XXXNodeTreeContext *context, const SocketRef *socket);
  XXXSocket(const XXXInputSocket &input_socket);
  XXXSocket(const XXXOutputSocket &output_socket);

  const XXXNodeTreeContext *context() const;
  const SocketRef *socket_ref() const;
  const SocketRef *operator->() const;

  friend bool operator==(const XXXSocket &a, const XXXSocket &b);
  friend bool operator!=(const XXXSocket &a, const XXXSocket &b);
  operator bool() const;

  uint64_t hash() const;
};

class XXXInputSocket : public XXXSocket {
 public:
  XXXInputSocket() = default;
  XXXInputSocket(const XXXNodeTreeContext *context, const InputSocketRef *socket);
  explicit XXXInputSocket(const XXXSocket &base_socket);

  const InputSocketRef *socket_ref() const;
  const InputSocketRef *operator->() const;

  XXXOutputSocket get_corresponding_group_node_output() const;
  XXXOutputSocket get_corresponding_group_input_socket() const;

  void foreach_origin_socket(FunctionRef<void(XXXSocket)> callback) const;
};

class XXXOutputSocket : public XXXSocket {
 public:
  XXXOutputSocket() = default;
  XXXOutputSocket(const XXXNodeTreeContext *context, const OutputSocketRef *socket);
  explicit XXXOutputSocket(const XXXSocket &base_socket);

  const OutputSocketRef *socket_ref() const;
  const OutputSocketRef *operator->() const;

  XXXInputSocket get_corresponding_group_node_input() const;
  XXXInputSocket get_corresponding_group_output_socket() const;

  void foreach_target_socket(FunctionRef<void(XXXInputSocket)> callback) const;
};

class XXXNodeTree {
 private:
  LinearAllocator<> allocator_;
  XXXNodeTreeContext *root_context_;
  VectorSet<const NodeTreeRef *> used_node_tree_refs_;

 public:
  XXXNodeTree(bNodeTree &btree, NodeTreeRefMap &node_tree_refs);
  ~XXXNodeTree();

  const XXXNodeTreeContext &root_context() const;
  Span<const NodeTreeRef *> used_node_tree_refs() const;

  bool has_link_cycles() const;
  void foreach_node(FunctionRef<void(XXXNode)> callback) const;

 private:
  XXXNodeTreeContext &construct_context_recursively(XXXNodeTreeContext *parent_context,
                                                    const NodeRef *parent_node,
                                                    bNodeTree &btree,
                                                    NodeTreeRefMap &node_tree_refs);
  void destruct_context_recursively(XXXNodeTreeContext *context);

  void foreach_node_in_context_recursive(const XXXNodeTreeContext &context,
                                         FunctionRef<void(XXXNode)> callback) const;
};

namespace xxx_node_tree_types {
using namespace node_tree_ref_types;
using nodes::XXXInputSocket;
using nodes::XXXNode;
using nodes::XXXNodeTree;
using nodes::XXXNodeTreeContext;
using nodes::XXXOutputSocket;
using nodes::XXXSocket;
}  // namespace xxx_node_tree_types

/* --------------------------------------------------------------------
 * XXXNodeTreeContext inline methods.
 */

inline const NodeTreeRef &XXXNodeTreeContext::tree() const
{
  return *tree_;
}

inline const XXXNodeTreeContext *XXXNodeTreeContext::parent_context() const
{
  return parent_context_;
}

inline const NodeRef *XXXNodeTreeContext::parent_node() const
{
  return parent_node_;
}

inline const XXXNodeTreeContext *XXXNodeTreeContext::child_context(const NodeRef &node) const
{
  return children_.lookup_default(&node, nullptr);
}

inline bool XXXNodeTreeContext::is_root() const
{
  return parent_context_ == nullptr;
}

/* --------------------------------------------------------------------
 * XXXNode inline methods.
 */

inline XXXNode::XXXNode(const XXXNodeTreeContext *context, const NodeRef *node_ref)
    : context_(context), node_ref_(node_ref)
{
  BLI_assert(node_ref == nullptr || &node_ref->tree() == &context->tree());
}

inline const XXXNodeTreeContext *XXXNode::context() const
{
  return context_;
}

inline const NodeRef *XXXNode::node_ref() const
{
  return node_ref_;
}

inline bool operator==(const XXXNode &a, const XXXNode &b)
{
  return a.context_ == b.context_ && a.node_ref_ == b.node_ref_;
}

inline bool operator!=(const XXXNode &a, const XXXNode &b)
{
  return !(a == b);
}

inline XXXNode::operator bool() const
{
  return node_ref_ != nullptr;
}

inline const NodeRef *XXXNode::operator->() const
{
  return node_ref_;
}

inline uint64_t XXXNode::hash() const
{
  return DefaultHash<const XXXNodeTreeContext *>{}(context_) ^
         DefaultHash<const NodeRef *>{}(node_ref_);
}

/* --------------------------------------------------------------------
 * XXXSocket inline methods.
 */

inline XXXSocket::XXXSocket(const XXXNodeTreeContext *context, const SocketRef *socket_ref)
    : context_(context), socket_ref_(socket_ref)
{
  BLI_assert(socket_ref == nullptr || &socket_ref->tree() == &context->tree());
}

inline XXXSocket::XXXSocket(const XXXInputSocket &input_socket)
    : XXXSocket(input_socket.context_, input_socket.socket_ref_)
{
}

inline XXXSocket::XXXSocket(const XXXOutputSocket &output_socket)
    : XXXSocket(output_socket.context_, output_socket.socket_ref_)
{
}

inline const XXXNodeTreeContext *XXXSocket::context() const
{
  return context_;
}

inline const SocketRef *XXXSocket::socket_ref() const
{
  return socket_ref_;
}

inline bool operator==(const XXXSocket &a, const XXXSocket &b)
{
  return a.context_ == b.context_ && a.socket_ref_ == b.socket_ref_;
}

inline bool operator!=(const XXXSocket &a, const XXXSocket &b)
{
  return !(a == b);
}

inline XXXSocket::operator bool() const
{
  return socket_ref_ != nullptr;
}

inline const SocketRef *XXXSocket::operator->() const
{
  return socket_ref_;
}

inline uint64_t XXXSocket::hash() const
{
  return DefaultHash<const XXXNodeTreeContext *>{}(context_) ^
         DefaultHash<const SocketRef *>{}(socket_ref_);
}

/* --------------------------------------------------------------------
 * XXXInputSocket inline methods.
 */

inline XXXInputSocket::XXXInputSocket(const XXXNodeTreeContext *context,
                                      const InputSocketRef *socket_ref)
    : XXXSocket(context, socket_ref)
{
}

inline XXXInputSocket::XXXInputSocket(const XXXSocket &base_socket) : XXXSocket(base_socket)
{
  BLI_assert(base_socket->is_input());
}

inline const InputSocketRef *XXXInputSocket::socket_ref() const
{
  return (const InputSocketRef *)socket_ref_;
}

inline const InputSocketRef *XXXInputSocket::operator->() const
{
  return (const InputSocketRef *)socket_ref_;
}

/* --------------------------------------------------------------------
 * XXXOutputSocket inline methods.
 */

inline XXXOutputSocket::XXXOutputSocket(const XXXNodeTreeContext *context,
                                        const OutputSocketRef *socket_ref)
    : XXXSocket(context, socket_ref)
{
}

inline XXXOutputSocket::XXXOutputSocket(const XXXSocket &base_socket) : XXXSocket(base_socket)
{
  BLI_assert(base_socket->is_output());
}

inline const OutputSocketRef *XXXOutputSocket::socket_ref() const
{
  return (const OutputSocketRef *)socket_ref_;
}

inline const OutputSocketRef *XXXOutputSocket::operator->() const
{
  return (const OutputSocketRef *)socket_ref_;
}

/* --------------------------------------------------------------------
 * XXXNodeTree inline methods.
 */

inline const XXXNodeTreeContext &XXXNodeTree::root_context() const
{
  return *root_context_;
}

inline Span<const NodeTreeRef *> XXXNodeTree::used_node_tree_refs() const
{
  return used_node_tree_refs_;
}

}  // namespace blender::nodes
