/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#pragma once

#include <memory>

#include "BLI_map.hh"
#include "BLI_string_ref.hh"
#include "BLI_vector.hh"

#include "DNA_node_types.h"

#include "NOD_derived_node_tree.hh"

#include "VPC_context.hh"
#include "VPC_operation.hh"
#include "VPC_result.hh"

namespace blender::viewport_compositor {

using namespace nodes::derived_node_tree_types;

/* The operation class that nodes should implement and instantiate in the bNodeType
 * get_compositor_operation, passing the given inputs to the constructor.  */
class NodeOperation : public Operation {
 private:
  /* The node that this operation represents. */
  DNode node_;

 public:
  /* Populate the output results based on the node outputs and populate the input descriptors based
   * on the node inputs. */
  NodeOperation(Context &context, DNode node);

  /* Returns a reference to the derived node that this operation represents. */
  const DNode &node() const;

  /* Returns a reference to the node that this operation represents. */
  const bNode &bnode() const;

 protected:
  /* Returns true if the output identified by the given identifier is needed and should be
   * computed, otherwise returns false. */
  bool is_output_needed(StringRef identifier) const;
};

}  // namespace blender::viewport_compositor
