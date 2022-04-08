/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#include <memory>

#include "BLI_map.hh"
#include "BLI_string_ref.hh"
#include "BLI_vector.hh"

#include "DNA_node_types.h"

#include "NOD_derived_node_tree.hh"
#include "NOD_node_declaration.hh"

#include "VPC_context.hh"
#include "VPC_node_operation.hh"
#include "VPC_operation.hh"
#include "VPC_result.hh"
#include "VPC_utilities.hh"

namespace blender::viewport_compositor {

using namespace nodes::derived_node_tree_types;

NodeOperation::NodeOperation(Context &context, DNode node) : Operation(context), node_(node)
{
  /* Populate the output results. */
  for (const OutputSocketRef *output : node->outputs()) {
    const ResultType result_type = get_node_socket_result_type(output);
    const Result result = Result(result_type, texture_pool());
    populate_result(output->identifier(), result);
  }

  /* Populate the input descriptors. */
  for (const InputSocketRef *input : node->inputs()) {
    InputDescriptor input_descriptor;
    input_descriptor.type = get_node_socket_result_type(input);
    const nodes::SocketDeclarationPtr &socket_declaration =
        input->node().declaration()->inputs()[input->index()];
    input_descriptor.domain_priority = socket_declaration->compositor_domain_priority();
    input_descriptor.expects_single_value = socket_declaration->compositor_expects_single_value();
    declare_input_descriptor(input->identifier(), input_descriptor);
  }

  populate_results_for_unlinked_inputs();
}

const bNode &NodeOperation::node() const
{
  return *node_->bnode();
}

bool NodeOperation::is_output_needed(StringRef identifier) const
{
  DOutputSocket output = node_.output_by_identifier(identifier);
  if (output->logically_linked_sockets().is_empty()) {
    return false;
  }
  return true;
}

void NodeOperation::pre_execute()
{
  /* For each unlinked input socket, allocate a single value and set the value to the socket's
   * default value. */
  for (const Map<StringRef, DInputSocket>::Item &item : unlinked_inputs_sockets_.items()) {
    Result &result = get_input(item.key);
    DInputSocket input = item.value;
    result.allocate_single_value();
    switch (result.type()) {
      case ResultType::Float:
        result.set_float_value(input->default_value<bNodeSocketValueFloat>()->value);
        continue;
      case ResultType::Vector:
        result.set_vector_value(float3(input->default_value<bNodeSocketValueVector>()->value));
        continue;
      case ResultType::Color:
        result.set_color_value(float4(input->default_value<bNodeSocketValueRGBA>()->value));
        continue;
    }
  }
}

void NodeOperation::populate_results_for_unlinked_inputs()
{
  for (const InputSocketRef *input_ref : node_->inputs()) {
    const DInputSocket input{node_.context(), input_ref};
    DSocket origin = get_node_input_origin_socket(input);

    /* Input is linked, skip it. If the origin is an input, that means the input is connected to an
     * unlinked input of a group input node, hence why we check if the origin is an output. */
    if (origin->is_output()) {
      continue;
    }

    /* Construct a result of an appropriate type, add it to the results vector, and map the input
     * to it. */
    const ResultType result_type = get_node_socket_result_type(origin.socket_ref());
    unlinked_inputs_results_.append(std::make_unique<Result>(result_type, texture_pool()));
    map_input_to_result(input->identifier(), unlinked_inputs_results_.last().get());

    /* Map the input to the socket to later allocate and initialize its value. */
    const DInputSocket origin_input{origin.context(), &origin->as_input()};
    unlinked_inputs_sockets_.add_new(input->identifier(), origin_input);
  }
}

}  // namespace blender::viewport_compositor
