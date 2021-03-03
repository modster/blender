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

#include "NOD_node_tree_ref.hh"

namespace blender::nodes {

class XXXNodeTreeContextInfo;
class XXXNodeTreeContext;
class XXXNodeTree;

struct XXXNode;
struct XXXSocket;
struct XXXInputSocket;
struct XXXOutputSocket;

class XXXNodeTreeContextInfo {
 private:
  XXXNodeTreeContextInfo *parent_;
  const NodeTreeRef *tree_;
  Map<const NodeRef *, XXXNodeTreeContextInfo *> children_;

  friend XXXNodeTree;

 public:
  const NodeTreeRef &tree() const;
};

class XXXNodeTreeContext {
 private:
  const XXXNodeTreeContextInfo *context_info_ = nullptr;

  friend XXXNodeTree;

 public:
  XXXNodeTreeContext() = default;
  XXXNodeTreeContext(const XXXNodeTreeContextInfo *context_info);

  friend bool operator==(const XXXNodeTreeContext &a, const XXXNodeTreeContext &b);
  friend bool operator!=(const XXXNodeTreeContext &a, const XXXNodeTreeContext &b);

  uint64_t hash() const;

  const XXXNodeTreeContextInfo &info() const;
};

struct XXXNode {
  XXXNodeTreeContext context;
  const NodeRef *node = nullptr;

  XXXNode() = default;
  XXXNode(XXXNodeTreeContext context, const NodeRef *node);

  friend bool operator==(const XXXNode &a, const XXXNode &b);
  friend bool operator!=(const XXXNode &a, const XXXNode &b);

  operator bool() const;

  uint64_t hash() const;
};

struct XXXSocket {
  XXXNodeTreeContext context;
  const SocketRef *socket;

  XXXSocket() = default;
  XXXSocket(XXXNodeTreeContext context, const SocketRef *socket);
  XXXSocket(const XXXInputSocket &input_socket);
  XXXSocket(const XXXOutputSocket &output_socket);

  friend bool operator==(const XXXSocket &a, const XXXSocket &b);
  friend bool operator!=(const XXXSocket &a, const XXXSocket &b);

  operator bool() const;

  uint64_t hash() const;
};

struct XXXInputSocket {
  XXXNodeTreeContext context;
  const InputSocketRef *socket = nullptr;

  XXXInputSocket() = default;
  XXXInputSocket(XXXNodeTreeContext context, const InputSocketRef *socket);
  explicit XXXInputSocket(const XXXSocket &base_socket);

  friend bool operator==(const XXXInputSocket &a, const XXXInputSocket &b);
  friend bool operator!=(const XXXInputSocket &a, const XXXInputSocket &b);

  operator bool() const;

  uint64_t hash() const;

  XXXOutputSocket try_get_single_origin() const;
};

struct XXXOutputSocket {
  XXXNodeTreeContext context;
  const OutputSocketRef *socket = nullptr;

  XXXOutputSocket() = default;
  XXXOutputSocket(XXXNodeTreeContext context, const OutputSocketRef *socket);
  explicit XXXOutputSocket(const XXXSocket &base_socket);

  friend bool operator==(const XXXOutputSocket &a, const XXXOutputSocket &b);
  friend bool operator!=(const XXXOutputSocket &a, const XXXOutputSocket &b);

  operator bool() const;

  uint64_t hash() const;
};

class XXXNodeTree {
 private:
  LinearAllocator<> allocator_;
  XXXNodeTreeContextInfo *root_context_info_;

 public:
  XXXNodeTree(bNodeTree &btree, NodeTreeRefMap &node_tree_refs);
  ~XXXNodeTree();

  const XXXNodeTreeContextInfo &root_context_info() const;

 private:
  XXXNodeTreeContextInfo &construct_context_info_recursively(XXXNodeTreeContextInfo *parent,
                                                             bNodeTree &btree,
                                                             NodeTreeRefMap &node_tree_refs);
  void destruct_context_info_recursively(XXXNodeTreeContextInfo *context_info);
};

namespace xxx_node_tree_types {
using nodes::XXXInputSocket;
using nodes::XXXNode;
using nodes::XXXNodeTree;
using nodes::XXXNodeTreeContext;
using nodes::XXXNodeTreeContextInfo;
using nodes::XXXOutputSocket;
using nodes::XXXSocket;
}  // namespace xxx_node_tree_types

/* --------------------------------------------------------------------
 * XXXNodeTreeContextInfo inline methods.
 */

inline const NodeTreeRef &XXXNodeTreeContextInfo::tree() const
{
  return *tree_;
}

/* --------------------------------------------------------------------
 * XXXNodeTreeContext inline methods.
 */

inline XXXNodeTreeContext::XXXNodeTreeContext(const XXXNodeTreeContextInfo *context_info)
    : context_info_(context_info)
{
}

inline bool operator==(const XXXNodeTreeContext &a, const XXXNodeTreeContext &b)
{
  return a.context_info_ == b.context_info_;
}

inline bool operator!=(const XXXNodeTreeContext &a, const XXXNodeTreeContext &b)
{
  return !(a == b);
}

inline uint64_t XXXNodeTreeContext::hash() const
{
  return DefaultHash<XXXNodeTreeContextInfo *>{}(context_info_);
}

inline const XXXNodeTreeContextInfo &XXXNodeTreeContext::info() const
{
  return *context_info_;
}

/* --------------------------------------------------------------------
 * XXXNode inline methods.
 */

inline XXXNode::XXXNode(XXXNodeTreeContext context, const NodeRef *node)
    : context(context), node(node)
{
}

inline bool operator==(const XXXNode &a, const XXXNode &b)
{
  return a.context == b.context && a.node == b.node;
}

inline bool operator!=(const XXXNode &a, const XXXNode &b)
{
  return !(a == b);
}

inline XXXNode::operator bool() const
{
  return node != nullptr;
}

inline uint64_t XXXNode::hash() const
{
  return DefaultHash<XXXNodeTreeContext>{}(context) ^ DefaultHash<const NodeRef *>{}(node);
}

/* --------------------------------------------------------------------
 * XXXSocket inline methods.
 */

inline XXXSocket::XXXSocket(XXXNodeTreeContext context, const SocketRef *socket)
    : context(context), socket(socket)
{
}

inline XXXSocket::XXXSocket(const XXXInputSocket &input_socket)
    : context(input_socket.context), socket(input_socket.socket)
{
}

inline XXXSocket::XXXSocket(const XXXOutputSocket &output_socket)
    : context(output_socket.context), socket(output_socket.socket)
{
}

inline bool operator==(const XXXSocket &a, const XXXSocket &b)
{
  return a.context == b.context && a.socket == b.socket;
}

inline bool operator!=(const XXXSocket &a, const XXXSocket &b)
{
  return !(a == b);
}

inline XXXSocket::operator bool() const
{
  return socket != nullptr;
}

inline uint64_t XXXSocket::hash() const
{
  return DefaultHash<XXXNodeTreeContext>{}(context) ^ DefaultHash<const SocketRef *>{}(socket);
}

/* --------------------------------------------------------------------
 * XXXInputSocket inline methods.
 */

inline XXXInputSocket::XXXInputSocket(XXXNodeTreeContext context, const InputSocketRef *socket)
    : context(context), socket(socket)
{
}

inline XXXInputSocket::XXXInputSocket(const XXXSocket &base_socket)
    : context(base_socket.context), socket(&base_socket.socket->as_input())
{
}

inline bool operator==(const XXXInputSocket &a, const XXXInputSocket &b)
{
  return a.context == b.context && a.socket == b.socket;
}

inline bool operator!=(const XXXInputSocket &a, const XXXInputSocket &b)
{
  return !(a == b);
}

inline XXXInputSocket::operator bool() const
{
  return socket != nullptr;
}

inline uint64_t XXXInputSocket::hash() const
{
  return DefaultHash<XXXNodeTreeContext>{}(context) ^
         DefaultHash<const InputSocketRef *>{}(socket);
}

/* --------------------------------------------------------------------
 * XXXOutputSocket inline methods.
 */

inline XXXOutputSocket::XXXOutputSocket(XXXNodeTreeContext context, const OutputSocketRef *socket)
    : context(context), socket(socket)
{
}

inline XXXOutputSocket::XXXOutputSocket(const XXXSocket &base_socket)
    : context(base_socket.context), socket(&base_socket.socket->as_output())
{
}

inline bool operator==(const XXXOutputSocket &a, const XXXOutputSocket &b)
{
  return a.context == b.context && a.socket == b.socket;
}

inline bool operator!=(const XXXOutputSocket &a, const XXXOutputSocket &b)
{
  return !(a == b);
}

inline XXXOutputSocket::operator bool() const
{
  return socket != nullptr;
}

inline uint64_t XXXOutputSocket::hash() const
{
  return DefaultHash<XXXNodeTreeContext>{}(context) ^
         DefaultHash<const OutputSocketRef *>{}(socket);
}

/* --------------------------------------------------------------------
 * XXXNodeTree inline methods.
 */

inline const XXXNodeTreeContextInfo &XXXNodeTree::root_context_info() const
{
  return *root_context_info_;
}

}  // namespace blender::nodes
