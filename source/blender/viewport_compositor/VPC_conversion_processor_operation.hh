/* SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include "GPU_shader.h"

#include "VPC_context.hh"
#include "VPC_input_descriptor.hh"
#include "VPC_processor_operation.hh"
#include "VPC_result.hh"

namespace blender::viewport_compositor {

/* -------------------------------------------------------------------------------------------------
 * Conversion Processor Operation
 *
 * A processor that converts a result from a certain type to another. See the derived classes for
 * more details. */
class ConversionProcessorOperation : public ProcessorOperation {
 public:
  using ProcessorOperation::ProcessorOperation;

  /* If the input result is a single value, execute_single is called. Otherwise, the shader
   * provided by get_conversion_shader is dispatched. */
  void execute() override;

  /* Determine if a conversion processor operation is needed for the input with the given result
   * and descriptor. If it is not needed, return a null pointer. If it is needed, return an
   * instance of the appropriate conversion processor. */
  static ProcessorOperation *construct_if_needed(Context &context,
                                                 const Result &input_result,
                                                 const InputDescriptor &input_descriptor);

 protected:
  /* Convert the input single value result to the output single value result. */
  virtual void execute_single(const Result &input, Result &output) = 0;

  /* Get the shader the will be used for conversion. */
  virtual GPUShader *get_conversion_shader() const = 0;
};

/* -------------------------------------------------------------------------------------------------
 * Convert Float To Vector Processor Operation
 *
 * Takes a float result and outputs a vector result. All three components of the output are filled
 * with the input float. */
class ConvertFloatToVectorProcessorOperation : public ConversionProcessorOperation {
 public:
  ConvertFloatToVectorProcessorOperation(Context &context);

  void execute_single(const Result &input, Result &output) override;

  GPUShader *get_conversion_shader() const override;
};

/* -------------------------------------------------------------------------------------------------
 * Convert Float To Color Processor Operation
 *
 * Takes a float result and outputs a color result. All three color channels of the output are
 * filled with the input float and the alpha channel is set to 1. */
class ConvertFloatToColorProcessorOperation : public ConversionProcessorOperation {
 public:
  ConvertFloatToColorProcessorOperation(Context &context);

  void execute_single(const Result &input, Result &output) override;

  GPUShader *get_conversion_shader() const override;
};

/* -------------------------------------------------------------------------------------------------
 * Convert Color To Float Processor Operation
 *
 * Takes a color result and outputs a float result. The output is the average of the three color
 * channels, the alpha channel is ignored. */
class ConvertColorToFloatProcessorOperation : public ConversionProcessorOperation {
 public:
  ConvertColorToFloatProcessorOperation(Context &context);

  void execute_single(const Result &input, Result &output) override;

  GPUShader *get_conversion_shader() const override;
};

/* -------------------------------------------------------------------------------------------------
 * Convert Color To Vector Processor Operation
 *
 * Takes a color result and outputs a vector result. The output is a copy of the three color
 * channels to the three vector components. */
class ConvertColorToVectorProcessorOperation : public ConversionProcessorOperation {
 public:
  ConvertColorToVectorProcessorOperation(Context &context);

  void execute_single(const Result &input, Result &output) override;

  GPUShader *get_conversion_shader() const override;
};

/* -------------------------------------------------------------------------------------------------
 * Convert Vector To Float Processor Operation
 *
 * Takes a vector result and outputs a float result. The output is the average of the three
 * components. */
class ConvertVectorToFloatProcessorOperation : public ConversionProcessorOperation {
 public:
  ConvertVectorToFloatProcessorOperation(Context &context);

  void execute_single(const Result &input, Result &output) override;

  GPUShader *get_conversion_shader() const override;
};

/* -------------------------------------------------------------------------------------------------
 * Convert Vector To Color Processor Operation
 *
 * Takes a vector result and outputs a color result. The output is a copy of the three vector
 * components to the three color channels with the alpha channel set to 1. */
class ConvertVectorToColorProcessorOperation : public ConversionProcessorOperation {
 public:
  ConvertVectorToColorProcessorOperation(Context &context);

  void execute_single(const Result &input, Result &output) override;

  GPUShader *get_conversion_shader() const override;
};

}  // namespace blender::viewport_compositor
