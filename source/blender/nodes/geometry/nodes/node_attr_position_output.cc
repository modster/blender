/*
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

#include "UI_interface.h"
#include "UI_resources.h"

#include "node_geometry_util.hh"

static bNodeSocketTemplate attr_node_position_output_in[] = {
    {SOCK_VECTOR, N_("Position"), 0.0f, 0.0f, 0.0f, 0.0f, -10000.0f, 10000.0f},
    {-1, ""},
};

void register_node_type_attr_position_output()
{
  static bNodeType ntype;

  attr_node_type_base(&ntype, ATTR_NODE_POSITION_OUTPUT, "Position Output", NODE_CLASS_OUTPUT, 0);
  node_type_socket_templates(&ntype, attr_node_position_output_in, nullptr);
  nodeRegisterType(&ntype);
}
