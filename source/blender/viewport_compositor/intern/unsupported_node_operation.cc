/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#include "VPC_result.hh"
#include "VPC_unsupported_node_operation.hh"

namespace blender::viewport_compositor {

using namespace nodes::derived_node_tree_types;

void UnsupportedNodeOperation::execute()
{
  for (const OutputSocketRef *output : node()->outputs()) {
    if (!is_output_needed(output->identifier())) {
      continue;
    }

    Result &result = get_result(output->identifier());
    result.allocate_single_value();
    switch (result.type()) {
      case ResultType::Float:
        result.set_float_value(0.0f);
      case ResultType::Vector:
        result.set_vector_value(float3(0.0f));
      case ResultType::Color:
        result.set_color_value(float4(0.0f));
    }
  }
}

}  // namespace blender::viewport_compositor
