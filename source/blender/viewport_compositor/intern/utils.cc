/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#include "NOD_derived_node_tree.hh"

namespace blender::viewport_compositor {

using namespace nodes::derived_node_tree_types;

DSocket get_node_input_origin_socket(DInputSocket input)
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

}  // namespace blender::viewport_compositor
