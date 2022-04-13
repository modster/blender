/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#pragma once

#include "NOD_derived_node_tree.hh"

#include "VPC_input_descriptor.hh"
#include "VPC_result.hh"

namespace blender::viewport_compositor {

using namespace nodes::derived_node_tree_types;

/* Get the origin socket of the given node input. If the input is not linked, the socket itself is
 * returned. If the input is linked, the socket that is linked to it is returned, which could
 * either be an input or an output. An input socket is returned when the given input is connected
 * to an unlinked input of a group input node. */
DSocket get_input_origin_socket(DInputSocket input);

/* Get the output socket linked to the given node input. If the input is not linked to an output, a
 * null output is returned. */
DOutputSocket get_output_linked_to_input(DInputSocket input);

/* Get the result type that corresponds to the type of the given socket. */
ResultType get_node_socket_result_type(const SocketRef *socket);

/* A node is a GPU material node if it defines a method to get a GPU material node operation. */
bool is_gpu_material_node(DNode node);

/* Get the input descriptor of the given input socket. */
InputDescriptor input_descriptor_from_input_socket(const InputSocketRef *socket);

}  // namespace blender::viewport_compositor
