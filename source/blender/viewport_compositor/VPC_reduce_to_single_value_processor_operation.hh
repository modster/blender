/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#pragma once

#include "VPC_context.hh"
#include "VPC_processor_operation.hh"
#include "VPC_result.hh"

namespace blender::viewport_compositor {

/* A processor that reduces its input result into a single value output result. The input is
 * assumed to be a texture result of size 1x1, that is, a texture composed of a single pixel, the
 * value of which shall serve as the single value of the output result. See
 * add_reduce_to_single_value_input_processor_if_needed. */
class ReduceToSingleValueProcessorOperation : public ProcessorOperation {
 public:
  ReduceToSingleValueProcessorOperation(Context &context, ResultType type);

  void execute() override;
};

}  // namespace blender::viewport_compositor
