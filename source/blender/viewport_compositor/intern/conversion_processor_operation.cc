/* SPDX-License-Identifier: GPL-2.0-or-later */

#include "BLI_math_vec_types.hh"

#include "GPU_shader.h"

#include "VPC_context.hh"
#include "VPC_conversion_processor_operation.hh"
#include "VPC_input_descriptor.hh"
#include "VPC_result.hh"
#include "VPC_utilities.hh"

namespace blender::viewport_compositor {

/* -------------------------------------------------------------------------------------------------
 *  Convert Processor Operation.
 */

void ConversionProcessorOperation::execute()
{
  Result &result = get_result();
  const Result &input = get_input();

  if (input.is_single_value()) {
    result.allocate_single_value();
    execute_single(input, result);
    return;
  }

  result.allocate_texture(input.domain());

  GPUShader *shader = get_conversion_shader();
  GPU_shader_bind(shader);

  input.bind_as_texture(shader, "input_sampler");
  result.bind_as_image(shader, "output_image");

  compute_dispatch_global(shader, input.domain().size);

  input.unbind_as_texture();
  result.unbind_as_image();
  GPU_shader_unbind();
}

ProcessorOperation *ConversionProcessorOperation::construct_if_needed(
    Context &context, const Result &input_result, const InputDescriptor &input_descriptor)
{
  ResultType result_type = input_result.type();
  ResultType expected_type = input_descriptor.type;

  /* If the result type differs from the expected type, return an instance of an appropriate
   * conversion processor. Otherwise, return a null pointer. */
  if (result_type == ResultType::Float && expected_type == ResultType::Vector) {
    return new ConvertFloatToVectorProcessorOperation(context);
  }
  else if (result_type == ResultType::Float && expected_type == ResultType::Color) {
    return new ConvertFloatToColorProcessorOperation(context);
  }
  else if (result_type == ResultType::Color && expected_type == ResultType::Float) {
    return new ConvertColorToFloatProcessorOperation(context);
  }
  else if (result_type == ResultType::Color && expected_type == ResultType::Vector) {
    return new ConvertColorToVectorProcessorOperation(context);
  }
  else if (result_type == ResultType::Vector && expected_type == ResultType::Float) {
    return new ConvertVectorToFloatProcessorOperation(context);
  }
  else if (result_type == ResultType::Vector && expected_type == ResultType::Color) {
    return new ConvertVectorToColorProcessorOperation(context);
  }
  else {
    return nullptr;
  }
}

/* -------------------------------------------------------------------------------------------------
 *  Convert Float To Vector Processor Operation.
 */

ConvertFloatToVectorProcessorOperation::ConvertFloatToVectorProcessorOperation(Context &context)
    : ConversionProcessorOperation(context)
{
  InputDescriptor input_descriptor;
  input_descriptor.type = ResultType::Float;
  declare_input_descriptor(input_descriptor);
  populate_result(Result(ResultType::Vector, texture_pool()));
}

void ConvertFloatToVectorProcessorOperation::execute_single(const Result &input, Result &output)
{
  output.set_vector_value(float3(input.get_float_value()));
}

GPUShader *ConvertFloatToVectorProcessorOperation::get_conversion_shader() const
{
  return shader_pool().acquire("compositor_convert_float_to_vector");
}

/* -------------------------------------------------------------------------------------------------
 *  Convert Float To Color Processor Operation.
 */

ConvertFloatToColorProcessorOperation::ConvertFloatToColorProcessorOperation(Context &context)
    : ConversionProcessorOperation(context)
{
  InputDescriptor input_descriptor;
  input_descriptor.type = ResultType::Float;
  declare_input_descriptor(input_descriptor);
  populate_result(Result(ResultType::Color, texture_pool()));
}

void ConvertFloatToColorProcessorOperation::execute_single(const Result &input, Result &output)
{
  float4 color = float4(input.get_float_value());
  color[3] = 1.0f;
  output.set_color_value(color);
}

GPUShader *ConvertFloatToColorProcessorOperation::get_conversion_shader() const
{
  return shader_pool().acquire("compositor_convert_float_to_color");
}

/* -------------------------------------------------------------------------------------------------
 *  Convert Color To Float Processor Operation.
 */

ConvertColorToFloatProcessorOperation::ConvertColorToFloatProcessorOperation(Context &context)
    : ConversionProcessorOperation(context)
{
  InputDescriptor input_descriptor;
  input_descriptor.type = ResultType::Color;
  declare_input_descriptor(input_descriptor);
  populate_result(Result(ResultType::Float, texture_pool()));
}

void ConvertColorToFloatProcessorOperation::execute_single(const Result &input, Result &output)
{
  float4 color = input.get_color_value();
  output.set_float_value((color[0] + color[1] + color[2]) / 3.0f);
}

GPUShader *ConvertColorToFloatProcessorOperation::get_conversion_shader() const
{
  return shader_pool().acquire("compositor_convert_color_to_float");
}

/* -------------------------------------------------------------------------------------------------
 *  Convert Color To Vector Processor Operation.
 */

ConvertColorToVectorProcessorOperation::ConvertColorToVectorProcessorOperation(Context &context)
    : ConversionProcessorOperation(context)
{
  InputDescriptor input_descriptor;
  input_descriptor.type = ResultType::Color;
  declare_input_descriptor(input_descriptor);
  populate_result(Result(ResultType::Vector, texture_pool()));
}

void ConvertColorToVectorProcessorOperation::execute_single(const Result &input, Result &output)
{
  float4 color = input.get_color_value();
  output.set_vector_value(float3(color));
}

GPUShader *ConvertColorToVectorProcessorOperation::get_conversion_shader() const
{
  return shader_pool().acquire("compositor_convert_color_to_vector");
}

/* -------------------------------------------------------------------------------------------------
 *  Convert Vector To Float Processor Operation.
 */

ConvertVectorToFloatProcessorOperation::ConvertVectorToFloatProcessorOperation(Context &context)
    : ConversionProcessorOperation(context)
{
  InputDescriptor input_descriptor;
  input_descriptor.type = ResultType::Vector;
  declare_input_descriptor(input_descriptor);
  populate_result(Result(ResultType::Float, texture_pool()));
}

void ConvertVectorToFloatProcessorOperation::execute_single(const Result &input, Result &output)
{
  float3 vector = input.get_vector_value();
  output.set_float_value((vector[0] + vector[1] + vector[2]) / 3.0f);
}

GPUShader *ConvertVectorToFloatProcessorOperation::get_conversion_shader() const
{
  return shader_pool().acquire("compositor_convert_vector_to_float");
}

/* -------------------------------------------------------------------------------------------------
 *  Convert Vector To Color Processor Operation.
 */

ConvertVectorToColorProcessorOperation::ConvertVectorToColorProcessorOperation(Context &context)
    : ConversionProcessorOperation(context)
{
  InputDescriptor input_descriptor;
  input_descriptor.type = ResultType::Vector;
  declare_input_descriptor(input_descriptor);
  populate_result(Result(ResultType::Color, texture_pool()));
}

void ConvertVectorToColorProcessorOperation::execute_single(const Result &input, Result &output)
{
  output.set_color_value(float4(input.get_vector_value(), 1.0f));
}

GPUShader *ConvertVectorToColorProcessorOperation::get_conversion_shader() const
{
  return shader_pool().acquire("compositor_convert_vector_to_color");
}

}  // namespace blender::viewport_compositor
