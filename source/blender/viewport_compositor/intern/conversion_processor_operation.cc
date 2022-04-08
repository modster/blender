/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#include "BLI_math_vec_types.hh"

#include "GPU_compute.h"
#include "GPU_shader.h"

#include "VPC_context.hh"
#include "VPC_conversion_processor_operation.hh"
#include "VPC_input_descriptor.hh"
#include "VPC_result.hh"

namespace blender::viewport_compositor {

const char *ConversionProcessorOperation::shader_input_sampler_name = "input_sampler";
const char *ConversionProcessorOperation::shader_output_image_name = "output_image";

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

  input.bind_as_texture(shader, shader_input_sampler_name);
  result.bind_as_image(shader, shader_output_image_name);

  const int2 size = result.domain().size;
  GPU_compute_dispatch(shader, size.x / 16 + 1, size.y / 16 + 1, 1);

  input.unbind_as_texture();
  result.unbind_as_image();
  GPU_shader_unbind();
  GPU_shader_free(shader);
}

/* --------------------------------------------------------------------
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

/* Use the shader for color conversion since they are stored in similar textures and the conversion
 * is practically the same. */
GPUShader *ConvertFloatToVectorProcessorOperation::get_conversion_shader() const
{
  return GPU_shader_create_from_info_name("compositor_convert_float_to_color");
}

/* --------------------------------------------------------------------
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
  return GPU_shader_create_from_info_name("compositor_convert_float_to_color");
}

/* --------------------------------------------------------------------
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
  return GPU_shader_create_from_info_name("compositor_convert_color_to_float");
}

/* --------------------------------------------------------------------
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

/* Use the shader for color conversion since they are stored in similar textures and the conversion
 * is practically the same. */
GPUShader *ConvertVectorToFloatProcessorOperation::get_conversion_shader() const
{
  return GPU_shader_create_from_info_name("compositor_convert_color_to_float");
}

/* --------------------------------------------------------------------
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
  return GPU_shader_create_from_info_name("compositor_convert_vector_to_color");
}

}  // namespace blender::viewport_compositor
