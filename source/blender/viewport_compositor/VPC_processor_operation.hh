/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#pragma once

#include "BLI_string_ref.hh"

#include "VPC_input_descriptor.hh"
#include "VPC_operation.hh"
#include "VPC_result.hh"

namespace blender::viewport_compositor {

/* A processor operation is an operation that takes exactly one input and computes exactly one
 * output. */
class ProcessorOperation : public Operation {
 public:
  /* The identifier of the output. This is constant for all operations. */
  static const StringRef output_identifier;
  /* The identifier of the input. This is constant for all operations. */
  static const StringRef input_identifier;

 public:
  using Operation::Operation;

  /* Get a reference to the output result of the processor, this essentially calls the super
   * get_result with the output identifier of the processor. */
  Result &get_result();

  /* Map the input of the processor to the given result, this essentially calls the super
   * map_input_to_result with the input identifier of the processor. */
  void map_input_to_result(Result *result);

 protected:
  /* Processor operations don't need input processors, so override with an empty implementation. */
  void evaluate_input_processors() override;

  /* Get a reference to the input result of the processor, this essentially calls the super
   * get_result with the input identifier of the processor. */
  Result &get_input();

  /* Switch the result mapped to the input with the given result, this essentially calls the super
   * switch_result_mapped_to_input with the input identifier of the processor. */
  void switch_result_mapped_to_input(Result *result);

  /* Populate the result of the processor, this essentially calls the super populate_result method
   * with the output identifier of the processor. */
  void populate_result(Result result);

  /* Declare the descriptor of the input of the processor to be the given descriptor, this
   * essentially calls the super declare_input_descriptor with the input identifier of the
   * processor. */
  void declare_input_descriptor(InputDescriptor descriptor);

  /* Get a reference to the descriptor of the input, this essentially calls the super
   * get_input_descriptor with the input identifier of the processor. */
  InputDescriptor &get_input_descriptor();
};

}  // namespace blender::viewport_compositor
