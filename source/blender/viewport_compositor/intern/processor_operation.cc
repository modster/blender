/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#include "VPC_input_descriptor.hh"
#include "VPC_operation.hh"
#include "VPC_processor_operation.hh"
#include "VPC_result.hh"

namespace blender::viewport_compositor {

const StringRef ProcessorOperation::input_identifier = StringRef("Input");
const StringRef ProcessorOperation::output_identifier = StringRef("Output");

Result &ProcessorOperation::get_result()
{
  return Operation::get_result(output_identifier);
}

void ProcessorOperation::map_input_to_result(Result *result)
{
  Operation::map_input_to_result(input_identifier, result);
}

void ProcessorOperation::evaluate_input_processors()
{
}

Result &ProcessorOperation::get_input()
{
  return Operation::get_input(input_identifier);
}

void ProcessorOperation::switch_result_mapped_to_input(Result *result)
{
  Operation::switch_result_mapped_to_input(input_identifier, result);
}

void ProcessorOperation::populate_result(Result result)
{
  Operation::populate_result(output_identifier, result);
}

void ProcessorOperation::declare_input_descriptor(InputDescriptor descriptor)
{
  Operation::declare_input_descriptor(input_identifier, descriptor);
}

InputDescriptor &ProcessorOperation::get_input_descriptor()
{
  return Operation::get_input_descriptor(input_identifier);
}

}  // namespace blender::viewport_compositor
