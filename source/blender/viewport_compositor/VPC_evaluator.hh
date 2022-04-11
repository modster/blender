/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#pragma once

#include "DNA_node_types.h"

#include "VPC_compiler.hh"
#include "VPC_context.hh"

namespace blender::viewport_compositor {

/* --------------------------------------------------------------------
 * Evaluator.
 */

/* The main class of the viewport compositor. The evaluator compiles the compositor node tree into
 * a stream of operations that are then executed to compute the output of the compositor. */
class Evaluator {
 private:
  /* The compiler instance used to compile the compositor node tree. */
  Compiler compiler_;

 public:
  Evaluator(Context &context, bNodeTree *node_tree);

  /* Compile the compositor node tree into an operations stream. */
  void compile();

  /* Evaluate the compiled operations stream. */
  void evaluate();
};

}  // namespace blender::viewport_compositor
