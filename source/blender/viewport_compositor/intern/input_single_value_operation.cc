/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#include "BLI_math_vec_types.hh"

#include "VPC_input_single_value_operation.hh"
#include "VPC_operation.hh"
#include "VPC_result.hh"
#include "VPC_utilities.hh"

namespace blender::viewport_compositor {

const StringRef InputSingleValueOperation::output_identifier_ = StringRef("Output");

InputSingleValueOperation::InputSingleValueOperation(Context &context, DInputSocket input_socket)
    : Operation(context), input_socket_(input_socket)
{
  /* Populate the output result. */
  const ResultType result_type = get_node_socket_result_type(input_socket_.socket_ref());
  const Result result = Result(result_type, texture_pool());
  populate_result(result);
}

void InputSingleValueOperation::execute()
{
  /* Allocate a single value for the result. */
  Result &result = get_result();
  result.allocate_single_value();

  /* Set the value of the result to the default value of the input socket. */
  switch (result.type()) {
    case ResultType::Float:
      result.set_float_value(input_socket_->default_value<bNodeSocketValueFloat>()->value);
      break;
    case ResultType::Vector:
      result.set_vector_value(
          float3(input_socket_->default_value<bNodeSocketValueVector>()->value));
      break;
    case ResultType::Color:
      result.set_color_value(float4(input_socket_->default_value<bNodeSocketValueRGBA>()->value));
      break;
  }
}

Result &InputSingleValueOperation::get_result()
{
  return Operation::get_result(output_identifier_);
}

void InputSingleValueOperation::populate_result(Result result)
{
  Operation::populate_result(output_identifier_, result);
}

}  // namespace blender::viewport_compositor
