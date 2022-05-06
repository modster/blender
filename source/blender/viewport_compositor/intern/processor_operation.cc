/* SPDX-License-Identifier: GPL-2.0-or-later */

#include "VPC_input_descriptor.hh"
#include "VPC_operation.hh"
#include "VPC_processor_operation.hh"
#include "VPC_result.hh"

namespace blender::viewport_compositor {

const StringRef ProcessorOperation::input_identifier_ = StringRef("Input");
const StringRef ProcessorOperation::output_identifier_ = StringRef("Output");

Result &ProcessorOperation::get_result()
{
  return Operation::get_result(output_identifier_);
}

void ProcessorOperation::map_input_to_result(Result *result)
{
  Operation::map_input_to_result(input_identifier_, result);
}

void ProcessorOperation::add_and_evaluate_input_processors()
{
}

Result &ProcessorOperation::get_input()
{
  return Operation::get_input(input_identifier_);
}

void ProcessorOperation::switch_result_mapped_to_input(Result *result)
{
  Operation::switch_result_mapped_to_input(input_identifier_, result);
}

void ProcessorOperation::populate_result(Result result)
{
  Operation::populate_result(output_identifier_, result);
  /* The result of a processor operation is guaranteed to have a single user. */
  get_result().set_initial_reference_count(1);
}

void ProcessorOperation::declare_input_descriptor(InputDescriptor descriptor)
{
  Operation::declare_input_descriptor(input_identifier_, descriptor);
}

InputDescriptor &ProcessorOperation::get_input_descriptor()
{
  return Operation::get_input_descriptor(input_identifier_);
}

}  // namespace blender::viewport_compositor
