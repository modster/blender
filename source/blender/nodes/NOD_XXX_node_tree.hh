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

class XXXNodeTree;

class XXXNodeTreeContextInfo {
 private:
  XXXNodeTreeContextInfo *parent_;
  NodeTreeRef *tree_;
  Map<const NodeRef *, XXXNodeTreeContextInfo *> children_;

  friend XXXNodeTree;
};

class XXXNodeTreeContext {
 private:
  XXXNodeTreeContextInfo *context_;

  friend XXXNodeTree;
};

struct XXXNode {
  XXXNodeTreeContext context;
  NodeRef *node;
};

struct XXXSocket {
  XXXNodeTreeContext context;
  SocketRef *socket;
};

struct XXXInputSocket {
  XXXNodeTreeContext context;
  InputSocketRef *socket;
};

struct XXXOutputSocket {
  XXXNodeTreeContext context;
  OutputSocketRef *socket;
};

class XXXNodeTree {
 private:
  LinearAllocator<> allocator_;
  XXXNodeTreeContextInfo *root_context_info_;

 public:
  XXXNodeTree(bNodeTree *btree, NodeTreeRefMap &node_tree_refs);
  ~XXXNodeTree();

 private:
  void destruct_context_info_recursively(XXXNodeTreeContextInfo *context_info);
};

}  // namespace blender::nodes
