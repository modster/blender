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

static bNodeSocketTemplate attr_node_set_attribute_in[] = {
    {SOCK_FLOAT, N_("Value"), 0.0, 0.0, 0.0, 0.0, -FLT_MAX, FLT_MAX},
    {SOCK_INT, N_("Value"), 0, 0, 0, 0, -100000, 100000},
    {SOCK_BOOLEAN, N_("Value")},
    {SOCK_VECTOR, N_("Value"), 0.0, 0.0, 0.0, 0.0, -FLT_MAX, FLT_MAX},
    {SOCK_RGBA, N_("Value"), 0.8, 0.8, 0.8, 1.0},
    {-1, ""},
};

void register_node_type_attr_set_attribute()
{
  static bNodeType ntype;

  attr_node_type_base(&ntype, ATTR_NODE_SET_ATTRIBUTE, "Set Attribute", NODE_CLASS_OUTPUT, 0);
  node_type_socket_templates(&ntype, attr_node_set_attribute_in, nullptr);
  nodeRegisterType(&ntype);
}
