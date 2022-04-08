/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#pragma once

#include "NOD_derived_node_tree.hh"

namespace blender::viewport_compositor {

using namespace nodes::derived_node_tree_types;

/* Get the origin socket of the given node input. If the input is not linked, the socket itself is
 * returned. If the input is linked, the socket that is linked to it is returned, which could
 * either be an input or an output. An input socket is returned when the given input is connected
 * to an unlinked input of a group input node. */
DSocket get_node_input_origin_socket(DInputSocket input);

}  // namespace blender::viewport_compositor
