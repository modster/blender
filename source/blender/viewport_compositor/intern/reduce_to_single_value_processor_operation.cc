/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#include "GPU_state.h"
#include "GPU_texture.h"

#include "MEM_guardedalloc.h"

#include "VPC_context.hh"
#include "VPC_input_descriptor.hh"
#include "VPC_reduce_to_single_value_processor_operation.hh"
#include "VPC_result.hh"

namespace blender::viewport_compositor {

ReduceToSingleValueProcessorOperation::ReduceToSingleValueProcessorOperation(Context &context,
                                                                             ResultType type)
    : ProcessorOperation(context)
{
  InputDescriptor input_descriptor;
  input_descriptor.type = type;
  declare_input_descriptor(input_descriptor);
  populate_result(Result(type, texture_pool()));
}

void ReduceToSingleValueProcessorOperation::execute()
{
  const Result &input = get_input();
  GPU_memory_barrier(GPU_BARRIER_TEXTURE_UPDATE);
  float *pixel = static_cast<float *>(GPU_texture_read(input.texture(), GPU_DATA_FLOAT, 0));

  Result &result = get_result();
  result.allocate_single_value();
  switch (result.type()) {
    case ResultType::Color:
      result.set_color_value(pixel);
      break;
    case ResultType::Vector:
      result.set_vector_value(pixel);
      break;
    case ResultType::Float:
      result.set_float_value(*pixel);
      break;
  }

  MEM_freeN(pixel);
}

}  // namespace blender::viewport_compositor
