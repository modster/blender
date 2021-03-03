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
};

class XXXNodeTreeContext {
 private:
  XXXNodeTreeContextInfo *context_ = nullptr;

  friend XXXNodeTree;

 public:
  friend bool operator==(const XXXNodeTreeContext &a, const XXXNodeTreeContext &b);
  friend bool operator!=(const XXXNodeTreeContext &a, const XXXNodeTreeContext &b);

  uint64_t hash() const;
};

struct XXXNode {
  XXXNodeTreeContext context;
  const NodeRef *node = nullptr;

  friend bool operator==(const XXXNode &a, const XXXNode &b);
  friend bool operator!=(const XXXNode &a, const XXXNode &b);

  uint64_t hash() const;
};

struct XXXSocket {
  XXXNodeTreeContext context;
  const SocketRef *socket;

  XXXSocket(const XXXInputSocket &input_socket);
  XXXSocket(const XXXOutputSocket &output_socket);

  friend bool operator==(const XXXSocket &a, const XXXSocket &b);
  friend bool operator!=(const XXXSocket &a, const XXXSocket &b);

  uint64_t hash() const;
};

struct XXXInputSocket {
  XXXNodeTreeContext context;
  const InputSocketRef *socket = nullptr;

  explicit XXXInputSocket(const XXXSocket &base_socket);

  friend bool operator==(const XXXInputSocket &a, const XXXInputSocket &b);
  friend bool operator!=(const XXXInputSocket &a, const XXXInputSocket &b);

  uint64_t hash() const;
};

struct XXXOutputSocket {
  XXXNodeTreeContext context;
  const OutputSocketRef *socket = nullptr;

  explicit XXXOutputSocket(const XXXSocket &base_socket);

  friend bool operator==(const XXXOutputSocket &a, const XXXOutputSocket &b);
  friend bool operator!=(const XXXOutputSocket &a, const XXXOutputSocket &b);

  uint64_t hash() const;
};

class XXXNodeTree {
 private:
  LinearAllocator<> allocator_;
  XXXNodeTreeContextInfo *root_context_info_;

 public:
  XXXNodeTree(bNodeTree &btree, NodeTreeRefMap &node_tree_refs);
  ~XXXNodeTree();

 private:
  XXXNodeTreeContextInfo &construct_context_info_recursively(XXXNodeTreeContextInfo *parent,
                                                             bNodeTree &btree,
                                                             NodeTreeRefMap &node_tree_refs);
  void destruct_context_info_recursively(XXXNodeTreeContextInfo *context_info);
};

/* --------------------------------------------------------------------
 * XXXNodeTreeContext inline methods.
 */

inline bool operator==(const XXXNodeTreeContext &a, const XXXNodeTreeContext &b)
{
  return a.context_ == b.context_;
}

inline bool operator!=(const XXXNodeTreeContext &a, const XXXNodeTreeContext &b)
{
  return !(a == b);
}

inline uint64_t XXXNodeTreeContext::hash() const
{
  return DefaultHash<XXXNodeTreeContextInfo *>{}(context_);
}

/* --------------------------------------------------------------------
 * XXXNode inline methods.
 */

inline bool operator==(const XXXNode &a, const XXXNode &b)
{
  return a.context == b.context && a.node == b.node;
}

inline bool operator!=(const XXXNode &a, const XXXNode &b)
{
  return !(a == b);
}

inline uint64_t XXXNode::hash() const
{
  return DefaultHash<XXXNodeTreeContext>{}(context) ^ DefaultHash<const NodeRef *>{}(node);
}

/* --------------------------------------------------------------------
 * XXXSocket inline methods.
 */

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

inline uint64_t XXXSocket::hash() const
{
  return DefaultHash<XXXNodeTreeContext>{}(context) ^ DefaultHash<const SocketRef *>{}(socket);
}

/* --------------------------------------------------------------------
 * XXXInputSocket inline methods.
 */

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

inline uint64_t XXXInputSocket::hash() const
{
  return DefaultHash<XXXNodeTreeContext>{}(context) ^
         DefaultHash<const InputSocketRef *>{}(socket);
}

/* --------------------------------------------------------------------
 * XXXOutputSocket inline methods.
 */

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

inline uint64_t XXXOutputSocket::hash() const
{
  return DefaultHash<XXXNodeTreeContext>{}(context) ^
         DefaultHash<const OutputSocketRef *>{}(socket);
}

}  // namespace blender::nodes
