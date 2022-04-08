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
  /* A vector storing unique pointers to the results mapped to unlinked inputs. */
  Vector<std::unique_ptr<Result>> unlinked_inputs_results_;
  /* A mapping between each unlinked input in the node identified by its identifier and its
   * corresponding input socket. */
  Map<StringRef, DInputSocket> unlinked_inputs_sockets_;

 public:
  /* Initialize members by the given arguments, populate the output results based on the node
   * outputs, populate the input types maps based on the node inputs, and add results for unlinked
   * inputs. */
  NodeOperation(Context &context, DNode node);

  /* Returns a reference to the node this operations represents. */
  const bNode &node() const;

 protected:
  /* Returns true if the output identified by the given identifier is needed and should be
   * computed, otherwise returns false. */
  bool is_output_needed(StringRef identifier) const;

  /* Set the values of the results for unlinked inputs. */
  void pre_execute() override;

 private:
  /* For each unlinked input in the node, construct a new result of an appropriate type, add it to
   * the unlinked_inputs_results_ vector, map the input to it, and map the input to its
   * corresponding input socket through the unlinked_inputs_sockets_ map. */
  void populate_results_for_unlinked_inputs();
};

}  // namespace blender::viewport_compositor
