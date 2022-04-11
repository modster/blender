/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#include "BLI_vector_set.hh"

#include "DNA_node_types.h"

#include "NOD_derived_node_tree.hh"

#include "VPC_compiler.hh"
#include "VPC_context.hh"
#include "VPC_gpu_material_node.hh"
#include "VPC_node_operation.hh"
#include "VPC_operation.hh"
#include "VPC_scheduler.hh"
#include "VPC_utilities.hh"

namespace blender::viewport_compositor {

using namespace nodes::derived_node_tree_types;

/* --------------------------------------------------------------------
 * GPU Material Compile Group.
 */

/* A node is a GPU material node if it defines a method to get a GPU material node operation. */
static bool is_gpu_material_node(DNode node)
{
  return node->typeinfo()->get_compositor_gpu_material_node;
}

void GPUMaterialCompileGroup::add(DNode node)
{
  sub_schedule_.add_new(node);
}

bool GPUMaterialCompileGroup::is_complete(DNode next_node)
{
  /* Sub schedule is empty, so the group is not complete. */
  if (sub_schedule_.is_empty()) {
    return false;
  }

  /* If the next node is not a GPU material node, then it can't be added to the group and the group
   * is considered complete. */
  if (!is_gpu_material_node(next_node)) {
    return true;
  }

  /* If the next node has inputs that are linked to nodes that are not part of this group, then it
   * can't be added to the group and the group is considered complete. */
  for (const InputSocketRef *input_ref : next_node->inputs()) {
    const DInputSocket input{next_node.context(), input_ref};
    DSocket origin = get_node_input_origin_socket(input);

    /* If the origin is an output, then the input is linked to the origin node. Check if the origin
     * node is not part of the group. */
    if (origin->is_output() && !sub_schedule_.contains(origin.node())) {
      return true;
    }
  }

  /* The next node can be added to the group, so it is not complete yet. */
  return false;
}

void GPUMaterialCompileGroup::reset()
{
  sub_schedule_.clear();
}

VectorSet<DNode> &GPUMaterialCompileGroup::get_sub_schedule()
{
  return sub_schedule_;
}

/* --------------------------------------------------------------------
 * Compiler.
 */

Compiler::Compiler(Context &context, bNodeTree *node_tree)
    : context_(context), tree_(*node_tree, tree_ref_map_)
{
}

Compiler::~Compiler()
{
  for (const Operation *operation : operations_stream_) {
    delete operation;
  }
}

void Compiler::compile()
{
  const Schedule schedule = compute_schedule(tree_);
  for (const DNode &node : schedule) {
    /* First check if the material compile group is complete, and if it is, compile it. */
    if (gpu_material_compile_group_.is_complete(node)) {
      compile_gpu_material_group();
    }

    /* If the node is a GPU material node, add it to the GPU material compile group, it will be
     * compiled later once the group is complete, see previous statement. */
    if (is_gpu_material_node(node)) {
      gpu_material_compile_group_.add(node);
      continue;
    }

    /* Otherwise, compile the node into a standard node operation. */
    compile_standard_node(node);
  }
}

OperationsStream &Compiler::operations_stream()
{
  return operations_stream_;
}

void Compiler::compile_standard_node(DNode node)
{
  /* Get an instance of the node's compositor operation and add it to both the operations stream
   * and the node operations map. This instance should be freed by the compiler when it is no
   * longer needed. */
  NodeOperation *operation = node->typeinfo()->get_compositor_operation(context_, node);
  operations_stream_.append(operation);
  node_operations_.add_new(node, operation);

  /* Map the inputs of the operation to the results of the outputs they are linked to. */
  map_node_operation_inputs_to_results(node, operation);
}

void Compiler::map_node_operation_inputs_to_results(DNode node, NodeOperation *operation)
{
  for (const InputSocketRef *input_ref : node->inputs()) {
    const DInputSocket input{node.context(), input_ref};

    /* Get the origin socket of this input, which will be an output socket if the input is linked
     * to an output. */
    DSocket origin = get_node_input_origin_socket(input);

    /* If the origin socket is an input, that means the input is unlinked. Unlinked inputs are
     * mapped internally to internal results, so skip here. */
    if (origin->is_input()) {
      continue;
    }

    /* Now that we know the origin is an output, construct a derived output from it. */
    const DOutputSocket output{origin.context(), &origin->as_output()};

    /* Map the input to the result we got from the output. */
    Result &result = get_output_socket_result(output);
    operation->map_input_to_result(input->identifier(), &result);
  }
}

void Compiler::compile_gpu_material_group()
{
  /* Get the sub schedule that is part of the GPU material group, instantiate a GPU Material
   * Operation from it, and add it to the operations stream. This instance should be freed by the
   * compiler when it is no longer needed. */
  Schedule &sub_schedule = gpu_material_compile_group_.get_sub_schedule();
  GPUMaterialOperation *operation = new GPUMaterialOperation(context_, sub_schedule);
  operations_stream_.append(operation);

  /* Map each of the nodes in the sub schedule to the compiled operation. */
  for (DNode node : sub_schedule) {
    gpu_material_operations_.add_new(node, operation);
  }

  /* Map the inputs of the operation to the results of the outputs they are linked to. */
  map_gpu_material_operation_inputs_to_results(operation);

  /* Reset the compile group to make it ready to track the next potential group. */
  gpu_material_compile_group_.reset();
}

void Compiler::map_gpu_material_operation_inputs_to_results(GPUMaterialOperation *operation)
{
  /* For each input of the operation, retrieve the result of the output linked to it, and map the
   * result to the input. */
  InputIdentifierToOutputSocketMap &map = operation->get_input_identifier_to_output_socket_map();
  for (const InputIdentifierToOutputSocketMap::Item &item : map.items()) {
    /* Map the input to the result we got from the output. */
    Result &result = get_output_socket_result(item.value);
    operation->map_input_to_result(item.key, &result);
  }
}

Result &Compiler::get_output_socket_result(DOutputSocket output)
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

}  // namespace blender::viewport_compositor
