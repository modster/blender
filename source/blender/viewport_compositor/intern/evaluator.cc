/* SPDX-License-Identifier: GPL-2.0-or-later */

#include "DNA_node_types.h"

#include "NOD_derived_node_tree.hh"

#include "VPC_compile_state.hh"
#include "VPC_context.hh"
#include "VPC_evaluator.hh"
#include "VPC_gpu_material_operation.hh"
#include "VPC_input_single_value_operation.hh"
#include "VPC_node_operation.hh"
#include "VPC_operation.hh"
#include "VPC_result.hh"
#include "VPC_scheduler.hh"
#include "VPC_utilities.hh"

namespace blender::viewport_compositor {

using namespace nodes::derived_node_tree_types;

Evaluator::Evaluator(Context &context, bNodeTree &node_tree)
    : context_(context), node_tree_(node_tree)
{
}

void Evaluator::evaluate()
{
  /* Reset the texture pool that was potentially populated from a previous evaluation. */
  context_.texture_pool().reset();

  /* The node tree is not compiled yet, so compile and evaluate the node tree. */
  if (!is_compiled_) {
    compile_and_evaluate();
    is_compiled_ = true;
    return;
  }

  /* The node tree is already compiled, so just go over the operations stream and evaluate the
   * operations in order. */
  for (const std::unique_ptr<Operation> &operation : operations_stream_) {
    operation->evaluate();
  }
}

void Evaluator::reset()
{
  /* Reset evaluator state. */
  operations_stream_.clear();
  derived_node_tree_.reset();
  node_tree_reference_map_.clear();

  /* Mark the node tree as in need to be compiled. */
  is_compiled_ = false;
}

void Evaluator::compile_and_evaluate()
{
  /* Construct and initialize a derived node tree from the compositor node tree. */
  derived_node_tree_.reset(new DerivedNodeTree(node_tree_, node_tree_reference_map_));

  /* Compute the node execution schedule. */
  const Schedule schedule = compute_schedule(*derived_node_tree_);

  /* Declare a compile state to use for tracking the state of the compilation. */
  CompileState compile_state(schedule);

  /* Go over the nodes in the schedule, compiling them into either node operations or GPU material
   * operations. */
  for (const DNode &node : schedule) {
    /* Ask the compile state if now would be a good time to compile the GPU material compile group
     * given the current node, and if it is, compile and evaluate it. */
    if (compile_state.should_compile_gpu_material_compile_group(node)) {
      compile_and_evaluate_gpu_material_compile_group(compile_state);
    }

    /* If the node is a GPU material node, then add it to the GPU material compile group. */
    if (is_gpu_material_node(node)) {
      compile_state.add_node_to_gpu_material_compile_group(node);
    }
    else {
      /* Otherwise, compile and evaluate the node into a node operation. */
      compile_and_evaluate_node(node, compile_state);
    }
  }
}

void Evaluator::compile_and_evaluate_node(DNode node, CompileState &compile_state)
{
  /* Get an instance of the node's compositor operation. */
  NodeOperation *operation = node->typeinfo()->get_compositor_operation(context_, node);

  /* Map the node to the compiled operation. */
  compile_state.map_node_to_node_operation(node, operation);

  /* Map the inputs of the operation to the results of the outputs they are linked to. */
  map_node_operation_inputs_to_their_results(node, operation, compile_state);

  /* Add the operation to the operations stream. This has to be done after input mapping because
   * the method may add Input Single Value Operations to the operations stream. */
  operations_stream_.append(std::unique_ptr<Operation>(operation));

  /* Compute the initial reference counts of the results of the operation. */
  operation->compute_results_reference_counts(compile_state.get_schedule());

  /* Evaluate the operation. */
  operation->evaluate();
}

void Evaluator::map_node_operation_inputs_to_their_results(DNode node,
                                                           NodeOperation *operation,
                                                           CompileState &compile_state)
{
  for (const InputSocketRef *input_ref : node->inputs()) {
    const DInputSocket input{node.context(), input_ref};

    /* Get the output linked to the input. */
    const DOutputSocket output = get_output_linked_to_input(input);

    /* The output is not null, which means the input is linked. So map the input to the result we
     * get from the output. */
    if (output) {
      Result &result = compile_state.get_result_from_output_socket(output);
      operation->map_input_to_result(input->identifier(), &result);
      continue;
    }

    /* Otherwise, the output is null, which means the input is unlinked. So map the input to the
     * result of a newly created Input Single Value Operation. */
    InputSingleValueOperation *input_operation = new InputSingleValueOperation(context_, input);
    operation->map_input_to_result(input->identifier(), &input_operation->get_result());

    /* Add the input operation to the operations stream. */
    operations_stream_.append(std::unique_ptr<InputSingleValueOperation>(input_operation));

    /* Evaluate the input operation. */
    input_operation->evaluate();
  }
}

void Evaluator::compile_and_evaluate_gpu_material_compile_group(CompileState &compile_state)
{
  /* Compile the GPU material compile group into a GPU Material Operation. */
  SubSchedule &sub_schedule = compile_state.get_gpu_material_compile_group_sub_schedule();
  GPUMaterialOperation *operation = new GPUMaterialOperation(context_, sub_schedule);

  /* Map each of the nodes in the sub-schedule to the compiled operation. */
  for (DNode node : sub_schedule) {
    compile_state.map_node_to_gpu_material_operation(node, operation);
  }

  /* Map the inputs of the operation to the results of the outputs they are linked to. */
  map_gpu_material_operation_inputs_to_their_results(operation, compile_state);

  /* Add the operation to the operations stream. */
  operations_stream_.append(std::unique_ptr<Operation>(operation));

  /* Compute the initial reference counts of the results of the operation. */
  operation->compute_results_reference_counts(compile_state.get_schedule());

  /* Evaluate the operation. */
  operation->evaluate();

  /* Clear the GPU material compile group to ready it for tracking the next GPU material. */
  compile_state.reset_gpu_material_compile_group();
}

void Evaluator::map_gpu_material_operation_inputs_to_their_results(GPUMaterialOperation *operation,
                                                                   CompileState &compile_state)
{
  /* For each input of the operation, retrieve the result of the output linked to it, and map the
   * result to the input. */
  InputsToLinkedOutputsMap &map = operation->get_inputs_to_linked_outputs_map();
  for (const InputsToLinkedOutputsMap::Item &item : map.items()) {
    Result &result = compile_state.get_result_from_output_socket(item.value);
    operation->map_input_to_result(item.key, &result);
  }
}

}  // namespace blender::viewport_compositor
