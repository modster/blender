/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#include "BLI_assert.h"
#include "BLI_math_vector.h"

#include "DNA_node_types.h"

#include "NOD_derived_node_tree.hh"

#include "GPU_material.h"

#include "VPC_gpu_material_node.hh"
#include "VPC_utilities.hh"

namespace blender::viewport_compositor {

using namespace nodes::derived_node_tree_types;

GPUMaterialNode::GPUMaterialNode(DNode node) : node_(node)
{
  populate_inputs();
  populate_outputs();
}

GPUNodeStack *GPUMaterialNode::get_inputs_array()
{
  return inputs_.data();
}

GPUNodeStack *GPUMaterialNode::get_outputs_array()
{
  return outputs_.data();
}

bNode &GPUMaterialNode::node() const
{
  return *node_->bnode();
}

static eGPUType gpu_type_from_socket_type(eNodeSocketDatatype type)
{
  switch (type) {
    case SOCK_FLOAT:
      return GPU_FLOAT;
    case SOCK_VECTOR:
      return GPU_VEC3;
    case SOCK_RGBA:
      return GPU_VEC4;
    default:
      BLI_assert_unreachable();
      return GPU_NONE;
  }
}

static void gpu_stack_vector_from_socket(float *vector, const SocketRef *socket)
{
  switch (socket->bsocket()->type) {
    case SOCK_FLOAT:
      vector[0] = socket->default_value<bNodeSocketValueFloat>()->value;
      return;
    case SOCK_VECTOR:
      copy_v3_v3(vector, socket->default_value<bNodeSocketValueVector>()->value);
      return;
    case SOCK_RGBA:
      copy_v4_v4(vector, socket->default_value<bNodeSocketValueRGBA>()->value);
      return;
    default:
      BLI_assert_unreachable();
  }
}

static void populate_gpu_node_stack(DSocket socket, GPUNodeStack &stack)
{
  /* Make sure this stack is not marked as the end of the stack array. */
  stack.end = false;
  /* This will be initialized later by the GPU material compiler or the compile method. */
  stack.link = nullptr;
  /* Socket type and its corresponding GPU type. */
  stack.sockettype = socket->bsocket()->type;
  stack.type = gpu_type_from_socket_type((eNodeSocketDatatype)socket->bsocket()->type);

  if (socket->is_input()) {
    /* Get the origin socket connected to the input if any. */
    const DInputSocket input{socket.context(), &socket->as_input()};
    DSocket origin = get_node_input_origin_socket(input);
    /* The input is linked if the origin socket is not null and is an output socket. Had it been an
     * input socket, then it is an unlinked input of a group input node. */
    stack.hasinput = origin->is_output();
    /* Get the socket value from the origin if it is an input, because then it would be an unlinked
     * input of a group input node, otherwise, get the value from the socket itself. */
    if (origin->is_input()) {
      gpu_stack_vector_from_socket(stack.vec, origin.socket_ref());
    }
    else {
      gpu_stack_vector_from_socket(stack.vec, socket.socket_ref());
    }
  }
  else {
    stack.hasoutput = socket->is_logically_linked();
    /* Populate the stack vector even for outputs because some nodes store their properties in the
     * default values of their outputs. */
    gpu_stack_vector_from_socket(stack.vec, socket.socket_ref());
  }
}

void GPUMaterialNode::populate_inputs()
{
  /* Reserve a stack for each input in addition to an extra stack at the end to mark the end of the
   * array, as this is what the GPU module functions expect. */
  inputs_.resize(node_->inputs().size() + 1);
  inputs_.last().end = true;

  for (int i = 0; i < node_->inputs().size(); i++) {
    populate_gpu_node_stack(node_.input(i), inputs_[i]);
  }
}

void GPUMaterialNode::populate_outputs()
{
  /* Reserve a stack for each output in addition to an extra stack at the end to mark the end of
   * the array, as this is what the GPU module functions expect. */
  outputs_.resize(node_->outputs().size() + 1);
  outputs_.last().end = true;

  for (int i = 0; i < node_->outputs().size(); i++) {
    populate_gpu_node_stack(node_.output(i), outputs_[i]);
  }
}

}  // namespace blender::viewport_compositor
