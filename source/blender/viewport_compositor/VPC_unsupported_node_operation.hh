/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#pragma once

#include "VPC_node_operation.hh"

namespace blender::viewport_compositor {

/* A node operation that sets all of its outputs to zero. This is used as a stub for nodes that are
 * not implemented yet.  */
class UnsupportedNodeOperation : public NodeOperation {
 public:
  using NodeOperation::NodeOperation;

  void execute() override;
};

}  // namespace blender::viewport_compositor
