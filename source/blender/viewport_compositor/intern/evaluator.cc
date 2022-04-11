/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#include "DNA_node_types.h"

#include "VPC_context.hh"
#include "VPC_evaluator.hh"
#include "VPC_operation.hh"

namespace blender::viewport_compositor {

Evaluator::Evaluator(Context &context, bNodeTree *node_tree) : compiler_(context, node_tree)
{
}

void Evaluator::compile()
{
  compiler_.compile();
}

void Evaluator::evaluate()
{
  for (Operation *operation : compiler_.operations_stream()) {
    operation->evaluate();
  }
}

}  // namespace blender::viewport_compositor
