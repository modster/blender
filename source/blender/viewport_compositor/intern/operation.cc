/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#include <limits>

#include "BLI_map.hh"
#include "BLI_string_ref.hh"
#include "BLI_vector.hh"

#include "VPC_context.hh"
#include "VPC_conversion_processor_operation.hh"
#include "VPC_domain.hh"
#include "VPC_input_descriptor.hh"
#include "VPC_processor_operation.hh"
#include "VPC_realize_on_domain_processor_operation.hh"
#include "VPC_reduce_to_single_value_processor_operation.hh"
#include "VPC_result.hh"
#include "VPC_texture_pool.hh"

namespace blender::viewport_compositor {

Operation::Operation(Context &context) : context_(context)
{
}

Operation::~Operation()
{
  for (const Vector<ProcessorOperation *> &processors : input_processors_.values()) {
    for (ProcessorOperation *processor : processors) {
      delete processor;
    }
  }
}

void Operation::evaluate()
{
  pre_execute();

  evaluate_input_processors();

  execute();

  release_inputs();
}

Result &Operation::get_result(StringRef identifier)
{
  return results_.lookup(identifier);
}

void Operation::map_input_to_result(StringRef identifier, Result *result)
{
  inputs_to_results_map_.add_new(identifier, result);
  result->increment_reference_count();
}

Domain Operation::compute_domain()
{
  /* In case no domain input was found, likely because all inputs are single values, then return an
   * identity domain. */
  Domain operation_domain = Domain::identity();
  int current_domain_priority = std::numeric_limits<int>::max();

  for (StringRef identifier : input_descriptors_.keys()) {
    const Result &result = get_input(identifier);
    const InputDescriptor &descriptor = get_input_descriptor(identifier);

    /* A single value input can't be a domain input. */
    if (result.is_single_value() || descriptor.expects_single_value) {
      continue;
    }

    /* Notice that the lower the domain priority value is, the higher the priority is, hence the
     * less than comparison. */
    if (descriptor.domain_priority < current_domain_priority) {
      operation_domain = result.domain();
      current_domain_priority = descriptor.domain_priority;
    }
  }

  return operation_domain;
}

void Operation::pre_execute()
{
}

void Operation::evaluate_input_processors()
{
  /* First, add all needed processors for each input. */
  for (const StringRef &identifier : inputs_to_results_map_.keys()) {
    add_reduce_to_single_value_input_processor_if_needed(identifier);
    add_implicit_conversion_input_processor_if_needed(identifier);
    add_realize_on_domain_input_processor_if_needed(identifier);
  }

  /* Then, switch the result mapped for each input of the operation to be that of the last
   * processor for that input if any input processor exist for it. */
  for (const StringRef &identifier : inputs_to_results_map_.keys()) {
    Vector<ProcessorOperation *> &processors = input_processors_.lookup_or_add_default(identifier);
    /* No input processors, nothing to do. */
    if (processors.is_empty()) {
      continue;
    }
    /* Replace the currently mapped result with the result of the last input processor. */
    switch_result_mapped_to_input(identifier, &processors.last()->get_result());
  }

  /* Finally, evaluate the input processors in order. */
  for (const Vector<ProcessorOperation *> &processors : input_processors_.values()) {
    for (ProcessorOperation *processor : processors) {
      processor->evaluate();
    }
  }
}

Result &Operation::get_input(StringRef identifier) const
{
  return *inputs_to_results_map_.lookup(identifier);
}

void Operation::switch_result_mapped_to_input(StringRef identifier, Result *result)
{
  get_input(identifier).release();
  inputs_to_results_map_.lookup(identifier) = result;
}

void Operation::populate_result(StringRef identifier, Result result)
{
  results_.add_new(identifier, result);
}

void Operation::declare_input_descriptor(StringRef identifier, InputDescriptor descriptor)
{
  input_descriptors_.add_new(identifier, descriptor);
}

InputDescriptor &Operation::get_input_descriptor(StringRef identifier)
{
  return input_descriptors_.lookup(identifier);
}

Context &Operation::context()
{
  return context_;
}

TexturePool &Operation::texture_pool()
{
  return context_.texture_pool();
}

void Operation::add_reduce_to_single_value_input_processor_if_needed(StringRef identifier)
{
  const Result &result = get_input(identifier);
  /* Input result is already a single value. */
  if (result.is_single_value()) {
    return;
  }

  /* The input is a full sized texture can can't be reduced to a single value. */
  if (result.domain().size != int2(1)) {
    return;
  }

  /* The input is a texture of a single pixel and can be reduced to a single value. */
  ProcessorOperation *processor = new ReduceToSingleValueProcessorOperation(context(),
                                                                            result.type());
  add_input_processor(identifier, processor);
}

void Operation::add_implicit_conversion_input_processor_if_needed(StringRef identifier)
{
  ResultType result_type = get_input(identifier).type();
  ResultType expected_type = input_descriptors_.lookup(identifier).type;

  if (result_type == ResultType::Float && expected_type == ResultType::Vector) {
    add_input_processor(identifier, new ConvertFloatToVectorProcessorOperation(context()));
  }
  else if (result_type == ResultType::Float && expected_type == ResultType::Color) {
    add_input_processor(identifier, new ConvertFloatToColorProcessorOperation(context()));
  }
  else if (result_type == ResultType::Color && expected_type == ResultType::Float) {
    add_input_processor(identifier, new ConvertColorToFloatProcessorOperation(context()));
  }
  else if (result_type == ResultType::Vector && expected_type == ResultType::Float) {
    add_input_processor(identifier, new ConvertVectorToFloatProcessorOperation(context()));
  }
  else if (result_type == ResultType::Vector && expected_type == ResultType::Color) {
    add_input_processor(identifier, new ConvertVectorToColorProcessorOperation(context()));
  }
}

void Operation::add_realize_on_domain_input_processor_if_needed(StringRef identifier)
{
  const InputDescriptor &descriptor = input_descriptors_.lookup(identifier);
  /* This input does not need realization. */
  if (descriptor.skip_realization) {
    return;
  }

  /* The input expects a single value and if no single value is provided, it will be ignored and a
   * default value will be used, so no need to realize it. */
  if (descriptor.expects_single_value) {
    return;
  }

  const Result &result = get_input(identifier);
  /* Input result is a single value and does not need realization. */
  if (result.is_single_value()) {
    return;
  }

  /* Input result only contains a single pixel and will be reduced to a single value result through
   * a ReduceToSingleValueProcessorOperation, so no need to realize it. */
  if (result.domain().size == int2(1)) {
    return;
  }

  /* The input have an identical domain to the operation domain, so no need to realize it. */
  if (result.domain() == compute_domain()) {
    return;
  }

  /* Realization is needed. */
  ProcessorOperation *processor = new RealizeOnDomainProcessorOperation(
      context(), compute_domain(), descriptor.type);
  add_input_processor(identifier, processor);
}

void Operation::add_input_processor(StringRef identifier, ProcessorOperation *processor)
{
  /* Get a reference to the input processors vector for the given input. */
  Vector<ProcessorOperation *> &processors = input_processors_.lookup_or_add_default(identifier);

  /* Get the result that should serve as the input for the processor. This is either the result
   * mapped to the input or the result of the last processor depending on whether this is the first
   * processor or not. */
  Result &result = processors.is_empty() ? get_input(identifier) : processors.last()->get_result();

  /* Map the input result of the processor and add it to the processors vector. No need to map the
   * result of the processor to the operation input as this will be done later in
   * evaluate_input_processors. */
  processor->map_input_to_result(&result);
  processors.append(processor);
}

void Operation::release_inputs()
{
  for (Result *result : inputs_to_results_map_.values()) {
    result->release();
  }
}

}  // namespace blender::viewport_compositor
