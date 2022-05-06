/* SPDX-License-Identifier: GPL-2.0-or-later */

#include "VPC_result.hh"
#include "VPC_unsupported_node_operation.hh"

namespace blender::viewport_compositor {

using namespace nodes::derived_node_tree_types;

void UnsupportedNodeOperation::execute()
{
  for (const OutputSocketRef *output : node()->outputs()) {
    if (!should_compute_output(output->identifier())) {
      continue;
    }
    Result &result = get_result(output->identifier());
    result.allocate_invalid();
  }
}

}  // namespace blender::viewport_compositor
