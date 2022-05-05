/* SPDX-License-Identifier: GPL-2.0-or-later */

#include <limits>

#include "BLI_math_vec_types.hh"

#include "DNA_node_types.h"

#include "NOD_derived_node_tree.hh"

#include "VPC_compile_state.hh"
#include "VPC_domain.hh"
#include "VPC_gpu_material_operation.hh"
#include "VPC_input_descriptor.hh"
#include "VPC_node_operation.hh"
#include "VPC_result.hh"
#include "VPC_utilities.hh"

namespace blender::viewport_compositor {

using namespace nodes::derived_node_tree_types;

void CompileState::add_node_to_gpu_material_compile_group(DNode node)
{
  /* Add the node to the GPU material compile group. */
  gpu_material_compile_group_.add_new(node);

  /* If the domain of the GPU material compile group is not yet determined or was determined to be
   * an identity domain, update it to be the computed domain of the node. */
  if (gpu_material_compile_group_domain_ == Domain::identity()) {
    gpu_material_compile_group_domain_ = compute_gpu_material_node_domain(node);
  }
}

SubSchedule &CompileState::get_gpu_material_compile_group_sub_schedule()
{
  return gpu_material_compile_group_;
}

void CompileState::reset_gpu_material_compile_group()
{
  return gpu_material_compile_group_.clear();
}

bool CompileState::should_compile_gpu_material_compile_group(DNode node)
{
  /* If the GPU material compile group is empty, then it can't be compiled yet. */
  if (gpu_material_compile_group_.is_empty()) {
    return false;
  }

  /* If the node is not a GPU material node, then it can't be added to the GPU material compile
   * group and the GPU material compile group is considered complete and should be compiled. */
  if (!is_gpu_material_node(node)) {
    return true;
  }

  /* If the computed domain of the node doesn't matches the domain of the GPU material compile
   * group, then it can't be added to the GPU material compile group and the GPU material compile
   * group is considered complete and should be compiled. Identity domains are an exception as they
   * are always compatible because they represents single values. */
  if (gpu_material_compile_group_domain_ != Domain::identity() &&
      gpu_material_compile_group_domain_ != compute_gpu_material_node_domain(node)) {
    return true;
  }

  /* Otherwise, the node is compatible and can be added to the compile group and it shouldn't be
   * compiled just yet. */
  return false;
}

void CompileState::map_node_to_node_operation(DNode node, NodeOperation *operations)
{
  return node_operations_.add_new(node, operations);
}

void CompileState::map_node_to_gpu_material_operation(DNode node, GPUMaterialOperation *operations)
{
  return gpu_material_operations_.add_new(node, operations);
}

Result &CompileState::get_result_from_output_socket(DOutputSocket output)
{
  /* The output belongs to a node that was compiled into a standard node operation, so return a
   * reference to the result from that operation using the output identifier. */
  if (node_operations_.contains(output.node())) {
    NodeOperation *operation = node_operations_.lookup(output.node());
    return operation->get_result(output->identifier());
  }

  /* Otherwise, the output belongs to a node that was compiled into a GPU material operation, so
   * retrieve the internal identifier of that output and return a reference to the result from
   * that operation using the retrieved identifier. */
  GPUMaterialOperation *operation = gpu_material_operations_.lookup(output.node());
  return operation->get_result(operation->get_output_identifier_from_output_socket(output));
}

Domain CompileState::compute_gpu_material_node_domain(DNode node)
{
  /* Default to an identity domain in case no domain input was found, most likely because all
   * inputs are single values. */
  Domain node_domain = Domain::identity();
  int current_domain_priority = std::numeric_limits<int>::max();

  /* Go over the inputs and find the domain of the non single value input with the highest domain
   * priority. */
  for (const InputSocketRef *input_ref : node->inputs()) {
    const DInputSocket input{node.context(), input_ref};

    /* Get the output linked to the input. If it is null, that means the input is unlinked, so skip
     * it. */
    const DOutputSocket output = get_output_linked_to_input(input);
    if (!output) {
      continue;
    }

    /* Get the input descriptor of the input. */
    const InputDescriptor input_descriptor = input_descriptor_from_input_socket(input_ref);

    /* If the output belongs to a node that is part of the GPU material compile group, then the
     * domain of the input is the domain of the compile group itself. */
    if (gpu_material_compile_group_.contains(output.node())) {
      /* Single value inputs can't be domain inputs. */
      if (gpu_material_compile_group_domain_.size == int2(1)) {
        continue;
      }

      /* Notice that the lower the domain priority value is, the higher the priority is, hence the
       * less than comparison. */
      if (input_descriptor.domain_priority < current_domain_priority) {
        node_domain = gpu_material_compile_group_domain_;
        current_domain_priority = input_descriptor.domain_priority;
      }
      continue;
    }

    /* Get the result linked to the input. */
    const Result &result = get_result_from_output_socket(output);

    /* A single value input can't be a domain input. */
    if (result.is_single_value() || input_descriptor.expects_single_value) {
      continue;
    }

    /* Notice that the lower the domain priority value is, the higher the priority is, hence the
     * less than comparison. */
    if (input_descriptor.domain_priority < current_domain_priority) {
      node_domain = result.domain();
      current_domain_priority = input_descriptor.domain_priority;
    }
  }

  return node_domain;
}

}  // namespace blender::viewport_compositor
