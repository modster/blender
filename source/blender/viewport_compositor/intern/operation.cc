/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#include <limits>
#include <memory>

#include "BLI_map.hh"
#include "BLI_string_ref.hh"
#include "BLI_vector.hh"

#include "VPC_context.hh"
#include "VPC_conversion_processor_operation.hh"
#include "VPC_domain.hh"
#include "VPC_input_descriptor.hh"
#include "VPC_operation.hh"
#include "VPC_processor_operation.hh"
#include "VPC_realize_on_domain_processor_operation.hh"
#include "VPC_reduce_to_single_value_processor_operation.hh"
#include "VPC_result.hh"
#include "VPC_texture_pool.hh"

namespace blender::viewport_compositor {

Operation::Operation(Context &context) : context_(context)
{
}

Operation::~Operation() = default;

void Operation::evaluate()
{
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
  results_mapped_to_inputs_.add_new(identifier, result);
  result->increment_reference_count();
}

Domain Operation::compute_domain()
{
  /* Default to an identity domain in case no domain input was found, most likely because all
   * inputs are single values. */
  Domain operation_domain = Domain::identity();
  int current_domain_priority = std::numeric_limits<int>::max();

  /* Go over the inputs and find the domain of the non single value input with the highest domain
   * priority. */
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

void Operation::add_and_evaluate_input_processors()
{
  /* Add and evaluate reduce to single value input processors if needed. */
  for (const StringRef &identifier : results_mapped_to_inputs_.keys()) {
    ProcessorOperation *single_value = ReduceToSingleValueProcessorOperation::construct_if_needed(
        context(), get_input(identifier));
    add_and_evaluate_input_processor(identifier, single_value);
  }

  /* Add and evaluate conversion input processors if needed. */
  for (const StringRef &identifier : results_mapped_to_inputs_.keys()) {
    ProcessorOperation *conversion = ConversionProcessorOperation::construct_if_needed(
        context(), get_input(identifier), get_input_descriptor(identifier));
    add_and_evaluate_input_processor(identifier, conversion);
  }

  /* Add and evaluate realize on domain input processors if needed. */
  for (const StringRef &identifier : results_mapped_to_inputs_.keys()) {
    ProcessorOperation *realize_on_domain = RealizeOnDomainProcessorOperation::construct_if_needed(
        context(), get_input(identifier), get_input_descriptor(identifier), compute_domain());
    add_and_evaluate_input_processor(identifier, realize_on_domain);
  }
}

void Operation::add_and_evaluate_input_processor(StringRef identifier,
                                                 ProcessorOperation *processor)
{
  /* Allow null inputs to facilitate construct_if_needed pattern of addition. For instance, see the
   * implementation of the add_and_evaluate_input_processors method. */
  if (!processor) {
    return;
  }

  /* Get a reference to the input processors vector for the given input. */
  ProcessorsVector &processors = input_processors_.lookup_or_add_default(identifier);

  /* Get the result that should serve as the input for the processor. This is either the result
   * mapped to the input or the result of the last processor depending on whether this is the first
   * processor or not. */
  Result &result = processors.is_empty() ? get_input(identifier) : processors.last()->get_result();

  /* Map the input result of the processor and add it to the processors vector. */
  processor->map_input_to_result(&result);
  processors.append(std::unique_ptr<ProcessorOperation>(processor));

  /* Switch the result mapped to the input to be the output result of the processor. */
  switch_result_mapped_to_input(identifier, &processor->get_result());

  /* Evaluate the input processor. */
  processor->evaluate();
}

Result &Operation::get_input(StringRef identifier) const
{
  return *results_mapped_to_inputs_.lookup(identifier);
}

void Operation::switch_result_mapped_to_input(StringRef identifier, Result *result)
{
  get_input(identifier).release();
  results_mapped_to_inputs_.lookup(identifier) = result;
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

void Operation::evaluate_input_processors()
{
  /* The input processors are not added yet, so add and evaluate the input processors. */
  if (!input_processors_added_) {
    add_and_evaluate_input_processors();
    input_processors_added_ = true;
    return;
  }

  /* The input processors are already added, so just go over the input processors and evaluate
   * them. */
  for (const ProcessorsVector &processors : input_processors_.values()) {
    for (const std::unique_ptr<ProcessorOperation> &processor : processors) {
      processor->evaluate();
    }
  }
}

void Operation::release_inputs()
{
  for (Result *result : results_mapped_to_inputs_.values()) {
    result->release();
  }
}

}  // namespace blender::viewport_compositor
