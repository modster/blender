/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#include "BLI_assert.h"

#include "DNA_node_types.h"

#include "NOD_derived_node_tree.hh"
#include "NOD_node_declaration.hh"

#include "VPC_operation.hh"
#include "VPC_result.hh"
#include "VPC_utilities.hh"

namespace blender::viewport_compositor {

using namespace nodes::derived_node_tree_types;

DSocket get_input_origin_socket(DInputSocket input)
{
  /* The input is unlinked. Return the socket itself. */
  if (input->logically_linked_sockets().is_empty()) {
    return input;
  }

  /* Only a single origin socket is guaranteed to exist. */
  DSocket socket;
  input.foreach_origin_socket([&](const DSocket origin) { socket = origin; });
  return socket;
}

DOutputSocket get_output_linked_to_input(DInputSocket input)
{
  /* Get the origin socket of this input, which will be an output socket if the input is linked
   * to an output. */
  const DSocket origin = get_input_origin_socket(input);

  /* If the origin socket is an input, that means the input is unlinked, return a null output
   * socket. */
  if (origin->is_input()) {
    return DOutputSocket();
  }

  /* Now that we know the origin is an output, return a derived output from it. */
  return DOutputSocket(origin.context(), &origin->as_output());
}

ResultType get_node_socket_result_type(const SocketRef *socket)
{
  switch (socket->bsocket()->type) {
    case SOCK_FLOAT:
      return ResultType::Float;
    case SOCK_VECTOR:
      return ResultType::Vector;
    case SOCK_RGBA:
      return ResultType::Color;
    default:
      BLI_assert_unreachable();
      return ResultType::Float;
  }
}

bool is_gpu_material_node(DNode node)
{
  return node->typeinfo()->get_compositor_gpu_material_node;
}

InputDescriptor input_descriptor_from_input_socket(const InputSocketRef *socket)
{
  using namespace nodes;
  InputDescriptor input_descriptor;
  input_descriptor.type = get_node_socket_result_type(socket);
  const NodeDeclaration *node_declaration = socket->node().declaration();
  const SocketDeclarationPtr &socket_declaration = node_declaration->inputs()[socket->index()];
  input_descriptor.domain_priority = socket_declaration->compositor_domain_priority();
  input_descriptor.expects_single_value = socket_declaration->compositor_expects_single_value();
  return input_descriptor;
}

}  // namespace blender::viewport_compositor
