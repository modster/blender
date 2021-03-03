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

struct XXXNode {
  const XXXNodeTreeContext *context = nullptr;
  const NodeRef *node_ref = nullptr;

  XXXNode() = default;
  XXXNode(const XXXNodeTreeContext *context, const NodeRef *node);

  friend bool operator==(const XXXNode &a, const XXXNode &b);
  friend bool operator!=(const XXXNode &a, const XXXNode &b);

  operator bool() const;
  const NodeRef *operator->() const;

  uint64_t hash() const;
};

struct XXXSocket {
  const XXXNodeTreeContext *context = nullptr;
  const SocketRef *socket_ref = nullptr;

  XXXSocket() = default;
  XXXSocket(const XXXNodeTreeContext *context, const SocketRef *socket);
  XXXSocket(const XXXInputSocket &input_socket);
  XXXSocket(const XXXOutputSocket &output_socket);

  friend bool operator==(const XXXSocket &a, const XXXSocket &b);
  friend bool operator!=(const XXXSocket &a, const XXXSocket &b);

  operator bool() const;
  const SocketRef *operator->() const;

  uint64_t hash() const;
};

struct XXXInputSocket {
  const XXXNodeTreeContext *context = nullptr;
  const InputSocketRef *socket_ref = nullptr;

  XXXInputSocket() = default;
  XXXInputSocket(const XXXNodeTreeContext *context, const InputSocketRef *socket);
  explicit XXXInputSocket(const XXXSocket &base_socket);

  friend bool operator==(const XXXInputSocket &a, const XXXInputSocket &b);
  friend bool operator!=(const XXXInputSocket &a, const XXXInputSocket &b);

  operator bool() const;
  const InputSocketRef *operator->() const;

  uint64_t hash() const;

  XXXOutputSocket get_corresponding_group_node_output() const;
  XXXOutputSocket get_corresponding_group_input_socket() const;
};

struct XXXOutputSocket {
  const XXXNodeTreeContext *context = nullptr;
  const OutputSocketRef *socket_ref = nullptr;

  XXXOutputSocket() = default;
  XXXOutputSocket(const XXXNodeTreeContext *context, const OutputSocketRef *socket);
  explicit XXXOutputSocket(const XXXSocket &base_socket);

  friend bool operator==(const XXXOutputSocket &a, const XXXOutputSocket &b);
  friend bool operator!=(const XXXOutputSocket &a, const XXXOutputSocket &b);

  operator bool() const;
  const OutputSocketRef *operator->() const;

  uint64_t hash() const;

  XXXInputSocket get_corresponding_group_node_input() const;
  XXXInputSocket get_corresponding_group_output_socket() const;
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

 private:
  XXXNodeTreeContext &construct_context_recursively(XXXNodeTreeContext *parent_context,
                                                    const NodeRef *parent_node,
                                                    bNodeTree &btree,
                                                    NodeTreeRefMap &node_tree_refs);
  void destruct_context_recursively(XXXNodeTreeContext *context);
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
    : context(context), node_ref(node_ref)
{
  BLI_assert(node_ref == nullptr || &node_ref->tree() == &context->tree());
}

inline bool operator==(const XXXNode &a, const XXXNode &b)
{
  return a.context == b.context && a.node_ref == b.node_ref;
}

inline bool operator!=(const XXXNode &a, const XXXNode &b)
{
  return !(a == b);
}

inline XXXNode::operator bool() const
{
  return node_ref != nullptr;
}

inline const NodeRef *XXXNode::operator->() const
{
  return node_ref;
}

inline uint64_t XXXNode::hash() const
{
  return DefaultHash<const XXXNodeTreeContext *>{}(context) ^
         DefaultHash<const NodeRef *>{}(node_ref);
}

/* --------------------------------------------------------------------
 * XXXSocket inline methods.
 */

inline XXXSocket::XXXSocket(const XXXNodeTreeContext *context, const SocketRef *socket_ref)
    : context(context), socket_ref(socket_ref)
{
  BLI_assert(socket_ref == nullptr || &socket_ref->tree() == &context->tree());
}

inline XXXSocket::XXXSocket(const XXXInputSocket &input_socket)
    : context(input_socket.context), socket_ref(input_socket.socket_ref)
{
}

inline XXXSocket::XXXSocket(const XXXOutputSocket &output_socket)
    : context(output_socket.context), socket_ref(output_socket.socket_ref)
{
}

inline bool operator==(const XXXSocket &a, const XXXSocket &b)
{
  return a.context == b.context && a.socket_ref == b.socket_ref;
}

inline bool operator!=(const XXXSocket &a, const XXXSocket &b)
{
  return !(a == b);
}

inline XXXSocket::operator bool() const
{
  return socket_ref != nullptr;
}

inline const SocketRef *XXXSocket::operator->() const
{
  return socket_ref;
}

inline uint64_t XXXSocket::hash() const
{
  return DefaultHash<const XXXNodeTreeContext *>{}(context) ^
         DefaultHash<const SocketRef *>{}(socket_ref);
}

/* --------------------------------------------------------------------
 * XXXInputSocket inline methods.
 */

inline XXXInputSocket::XXXInputSocket(const XXXNodeTreeContext *context,
                                      const InputSocketRef *socket_ref)
    : context(context), socket_ref(socket_ref)
{
  BLI_assert(socket_ref == nullptr || &socket_ref->tree() == &context->tree());
}

inline XXXInputSocket::XXXInputSocket(const XXXSocket &base_socket)
    : context(base_socket.context), socket_ref(&base_socket.socket_ref->as_input())
{
  BLI_assert(socket_ref == nullptr || &socket_ref->tree() == &context->tree());
}

inline bool operator==(const XXXInputSocket &a, const XXXInputSocket &b)
{
  return a.context == b.context && a.socket_ref == b.socket_ref;
}

inline bool operator!=(const XXXInputSocket &a, const XXXInputSocket &b)
{
  return !(a == b);
}

inline XXXInputSocket::operator bool() const
{
  return socket_ref != nullptr;
}

inline const InputSocketRef *XXXInputSocket::operator->() const
{
  return socket_ref;
}

inline uint64_t XXXInputSocket::hash() const
{
  return DefaultHash<const XXXNodeTreeContext *>{}(context) ^
         DefaultHash<const InputSocketRef *>{}(socket_ref);
}

/* --------------------------------------------------------------------
 * XXXOutputSocket inline methods.
 */

inline XXXOutputSocket::XXXOutputSocket(const XXXNodeTreeContext *context,
                                        const OutputSocketRef *socket_ref)
    : context(context), socket_ref(socket_ref)
{
}

inline XXXOutputSocket::XXXOutputSocket(const XXXSocket &base_socket)
    : context(base_socket.context), socket_ref(&base_socket.socket_ref->as_output())
{
}

inline bool operator==(const XXXOutputSocket &a, const XXXOutputSocket &b)
{
  return a.context == b.context && a.socket_ref == b.socket_ref;
}

inline bool operator!=(const XXXOutputSocket &a, const XXXOutputSocket &b)
{
  return !(a == b);
}

inline XXXOutputSocket::operator bool() const
{
  return socket_ref != nullptr;
}

inline const OutputSocketRef *XXXOutputSocket::operator->() const
{
  return socket_ref;
}

inline uint64_t XXXOutputSocket::hash() const
{
  return DefaultHash<const XXXNodeTreeContext *>{}(context) ^
         DefaultHash<const OutputSocketRef *>{}(socket_ref);
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
