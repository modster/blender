/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#pragma once

#include "GPU_shader.h"

#include "VPC_context.hh"
#include "VPC_processor_operation.hh"
#include "VPC_result.hh"

namespace blender::viewport_compositor {

/* --------------------------------------------------------------------
 *  Conversion Processor Operation.
 */

/* A conversion processor is a processor that converts a result from a certain type to another. See
   the derived classes for more details. */
class ConversionProcessorOperation : public ProcessorOperation {
 public:
  /* The name of the input sampler in the conversion shader.
   * This is constant for all operations. */
  static const char *shader_input_sampler_name;
  /* The name of the output image in the conversion shader.
   * This is constant for all operations. */
  static const char *shader_output_image_name;

 public:
  using ProcessorOperation::ProcessorOperation;

  void execute() override;

 protected:
  /* Convert the input single value result to the output single value result. */
  virtual void execute_single(const Result &input, Result &output) = 0;

  /* Get the shader the will be used for conversion. It should have an input sampler called
   * shader_input_sampler_name and an output image of an appropriate type called
   * shader_output_image_name. */
  virtual GPUShader *get_conversion_shader() const = 0;
};

/* --------------------------------------------------------------------
 *  Convert Float To Vector Processor Operation.
 */

/* Takes a float result and outputs a vector result. All three components of the output are filled
 * with the input float. */
class ConvertFloatToVectorProcessorOperation : public ConversionProcessorOperation {
 public:
  ConvertFloatToVectorProcessorOperation(Context &context);

  void execute_single(const Result &input, Result &output) override;

  GPUShader *get_conversion_shader() const override;
};

/* --------------------------------------------------------------------
 *  Convert Float To Color Processor Operation.
 */

/* Takes a float result and outputs a color result. All three color channels of the output are
 * filled with the input float and the alpha channel is set to 1. */
class ConvertFloatToColorProcessorOperation : public ConversionProcessorOperation {
 public:
  ConvertFloatToColorProcessorOperation(Context &context);

  void execute_single(const Result &input, Result &output) override;

  GPUShader *get_conversion_shader() const override;
};

/* --------------------------------------------------------------------
 *  Convert Color To Float Processor Operation.
 */

/* Takes a color result and outputs a float result. The output is the average of the three color
 * channels, the alpha channel is ignored. */
class ConvertColorToFloatProcessorOperation : public ConversionProcessorOperation {
 public:
  ConvertColorToFloatProcessorOperation(Context &context);

  void execute_single(const Result &input, Result &output) override;

  GPUShader *get_conversion_shader() const override;
};

/* --------------------------------------------------------------------
 *  Convert Vector To Float Processor Operation.
 */

/* Takes a vector result and outputs a float result. The output is the average of the three
 * components. */
class ConvertVectorToFloatProcessorOperation : public ConversionProcessorOperation {
 public:
  ConvertVectorToFloatProcessorOperation(Context &context);

  void execute_single(const Result &input, Result &output) override;

  GPUShader *get_conversion_shader() const override;
};

/* --------------------------------------------------------------------
 *  Convert Vector To Color Processor Operation.
 */

/* Takes a vector result and outputs a color result. The output is a copy of the three vector
 * components to the three color channels with the alpha channel set to 1. */
class ConvertVectorToColorProcessorOperation : public ConversionProcessorOperation {
 public:
  ConvertVectorToColorProcessorOperation(Context &context);

  void execute_single(const Result &input, Result &output) override;

  GPUShader *get_conversion_shader() const override;
};

}  // namespace blender::viewport_compositor
