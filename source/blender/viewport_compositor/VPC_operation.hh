/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#pragma once

#include "BLI_map.hh"
#include "BLI_string_ref.hh"
#include "BLI_vector.hh"

#include "VPC_context.hh"
#include "VPC_domain.hh"
#include "VPC_input_descriptor.hh"
#include "VPC_result.hh"
#include "VPC_texture_pool.hh"

namespace blender::viewport_compositor {

/* Forward declare processor operation because it is used in the operation definition.  */
class ProcessorOperation;

/* The most basic unit of the compositor. The class can be implemented to perform a certain action
 * in the compositor. */
class Operation {
 private:
  /* A reference to the compositor context. This member references the same object in all
   * operations but is included in the class for convenience. */
  Context &context_;
  /* A mapping between each output of the operation identified by its identifier and the computed
   * result for that output. A result for each output of an appropriate type should be constructed
   * and added to the map during operation construction. The results should be allocated and their
   * contents should be computed in the execute method. */
  Map<StringRef, Result> results_;
  /* A mapping between each input of the operation identified by its identifier and a reference to
   * the computed result providing its data. The mapped result can be one that was computed by
   * another operation or one that was internally computed in the operation as part of an internal
   * preprocessing step like implicit conversion. It is the responsibility of the evaluator to map
   * the inputs to their linked results prior to invoking any method, which is done by calling
   * map_input_to_result. It is the responsibility of the operation to map the inputs that are not
   * linked to the result of an internal single value result computed by the operation during
   * operation construction. */
  Map<StringRef, Result *> inputs_to_results_map_;
  /* A mapping between each input of the operation identified by its identifier and an ordered list
   * of input processor operations to be applied on that input. */
  Map<StringRef, Vector<ProcessorOperation *>> input_processors_;
  /* A mapping between each input of the operation identified by its identifier and its input
   * descriptor. This should be populated during operation construction. */
  Map<StringRef, InputDescriptor> input_descriptors_;

 public:
  Operation(Context &context);

  virtual ~Operation();

  /* Evaluate the operation as follows:
   * 1. Run any pre-execute computations.
   * 2. Add an evaluate any input processors.
   * 3. Invoking the execute method of the operation.
   * 4. Releasing the results mapped to the inputs. */
  void evaluate();

  /* Get a reference to the output result identified by the given identifier. */
  Result &get_result(StringRef identifier);

  /* Map the input identified by the given identifier to the result providing its data. This also
   * increments the reference count of the result. See inputs_to_results_map_ for more details.
   * This should be called by the evaluator to establish links between different operations. */
  void map_input_to_result(StringRef identifier, Result *result);

 protected:
  /* Compute the operation domain of this operation. By default, this implements a default logic
   * that infers the operation domain from the inputs, which may be overridden for a different
   * logic. See the Domain class for the inference logic and more information. */
  virtual Domain compute_domain();

  /* This method is called before the execute method and can be overridden by a derived class to do
   * any necessary internal computations before the operation is executed. For instance, this is
   * overridden by node operations to compute results for unlinked sockets. */
  virtual void pre_execute();

  /* First, all the necessary input processors for each input. Then update the result mapped to
   * each input to be that of the last processor for that input if any input processors exist for
   * it. This is done now in a separate step after all processors were added because the operation
   * might use the original mapped results to determine what processors needs to be added. Finally,
   * evaluate all input processors in order. This is called before executing the operation to
   * prepare its inputs but after the pre_execute method was called. The class defines a default
   * implementation, but derived class can override the method to have a different
   * implementation, extend the implementation, or remove it. */
  virtual void evaluate_input_processors();

  /* This method should allocate the operation results, execute the operation, and compute the
   * output results. */
  virtual void execute() = 0;

  /* Get a reference to the result connected to the input identified by the given identifier. */
  Result &get_input(StringRef identifier) const;

  /* Switch the result mapped to the input identified by the given identifier with the given
   * result. This will involve releasing the original result, but it is assumed that the result
   * will be mapped to something else. */
  void switch_result_mapped_to_input(StringRef identifier, Result *result);

  /* Add the given result to the results_ map identified by the given output identifier. This
   * should be called during operation construction for every output. The provided result shouldn't
   * be allocated or initialized, this will happen later during execution. */
  void populate_result(StringRef identifier, Result result);

  /* Declare the descriptor of the input identified by the given identifier to be the given
   * descriptor. Adds the given descriptor to the input_descriptors_ map identified by the given
   * input identifier. This should be called during operation constructor for every input. */
  void declare_input_descriptor(StringRef identifier, InputDescriptor descriptor);

  /* Get a reference to the descriptor of the input identified by the given identified. */
  InputDescriptor &get_input_descriptor(StringRef identified);

  /* Returns a reference to the compositor context. */
  Context &context();

  /* Returns a reference to the texture pool of the compositor context. */
  TexturePool &texture_pool();

 private:
  /* Add a reduce to single value input processor for the input identified by the given identifier
   * if needed. */
  void add_reduce_to_single_value_input_processor_if_needed(StringRef identifier);

  /* Add an implicit conversion input processor for the input identified by the given identifier if
   * needed. */
  void add_implicit_conversion_input_processor_if_needed(StringRef identifier);

  /* Add a realize on domain input processor for the input identified by the given identifier if
   * needed. See the Domain class for more information. */
  void add_realize_on_domain_input_processor_if_needed(StringRef identifier);

  /* Add the given input processor operation to the list of input processors for the input
   * identified by the given identifier. This will also involve mapping the input of the processor
   * to be the result of the last input processor or the result mapped to the input if no previous
   * processors exists. The result of the last input processor will not be mapped to the operation
   * input in this method, this will be done later, see evaluate_input_processors for more
   * information. */
  void add_input_processor(StringRef identifier, ProcessorOperation *processor);

  /* Release the results that are mapped to the inputs of the operation. This is called after the
   * evaluation of the operation to declare that the results are no longer needed by this
   * operation. */
  void release_inputs();
};

}  // namespace blender::viewport_compositor
