/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#pragma once

#include "GPU_shader.h"

#include "VPC_context.hh"
#include "VPC_domain.hh"
#include "VPC_input_descriptor.hh"
#include "VPC_processor_operation.hh"
#include "VPC_result.hh"

namespace blender::viewport_compositor {

/* A processor that projects the input on a certain target domain, copies the area of the input
 * that intersects the target domain, and fill the rest with zeros or repetitions of the input
 * depending on the realization options of the target domain. See the discussion in VPC_domain.hh
 * for more information. */
class RealizeOnDomainProcessorOperation : public ProcessorOperation {
 private:
  /* The target domain to realize the input on. */
  Domain domain_;

 public:
  RealizeOnDomainProcessorOperation(Context &context, Domain domain, ResultType type);

  void execute() override;

  /* Determine if a realize on domain processor operation is needed for the input with the
   * given result and descriptor in an operation with the given operation domain. If it is not
   * needed, return a null pointer. If it is needed, return an instance of the processor. */
  static ProcessorOperation *construct_if_needed(Context &context,
                                                 const Result &input_result,
                                                 const InputDescriptor &input_descriptor,
                                                 const Domain &operaiton_domain);

 protected:
  /* The operation domain is just the target domain. */
  Domain compute_domain() override;

 private:
  /* Get the realization shader of the appropriate type. */
  GPUShader *get_realization_shader();
};

}  // namespace blender::viewport_compositor
