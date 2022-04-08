/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#pragma once

#include "GPU_shader.h"

#include "VPC_context.hh"
#include "VPC_domain.hh"
#include "VPC_processor_operation.hh"
#include "VPC_result.hh"

namespace blender::viewport_compositor {

/* A realize on domain processor is a processor that projects the input on a certain domain, copies
 * the area of the input that intersects the target domain, and fill the rest with zeros. See the
 * Domain class for more information. */
class RealizeOnDomainProcessorOperation : public ProcessorOperation {
 private:
  /* The target domain to realize the input on. */
  Domain domain_;

 public:
  RealizeOnDomainProcessorOperation(Context &context, Domain domain, ResultType type);

  void execute() override;

 protected:
  /* The operation domain is just the target domain. */
  Domain compute_domain() override;

 private:
  /* Get the realization shader of the appropriate type. */
  GPUShader *get_realization_shader();
};

}  // namespace blender::viewport_compositor
