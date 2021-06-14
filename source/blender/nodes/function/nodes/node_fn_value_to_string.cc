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

#include "node_function_util.hh"
#include <iomanip>

static bNodeSocketTemplate fn_node_value_to_string_in[] = {
    {SOCK_FLOAT, N_("Value"), 0.0f, 0.0f, 0.0f, 0.0f, -FLT_MAX, FLT_MAX},
    {SOCK_INT, N_("Decimals"), 2, 0, 0, 0, 0, 10},
    {-1, ""},
};

static bNodeSocketTemplate fn_node_value_to_string_out[] = {
    {SOCK_STRING, N_("String")},
    {-1, ""},
};

static void fn_node_value_to_string_expand_in_mf_network(
    blender::nodes::NodeMFNetworkBuilder &builder)
{
  static blender::fn::CustomMF_SI_SI_SO<float, int, std::string> to_str_fn{
      "Value To String", [](float a, int b) {
        std::stringstream stream;
        stream << std::fixed << std::setprecision(b) << a;
        return stream.str();
      }};
  const blender::fn::MultiFunction &fn = to_str_fn;
  builder.set_matching_fn(fn);
}

void register_node_type_fn_value_to_string()
{
  static bNodeType ntype;

  fn_node_type_base(&ntype, FN_NODE_VALUE_TO_STRING, "Value To String", NODE_CLASS_CONVERTOR, 0);
  node_type_socket_templates(&ntype, fn_node_value_to_string_in, fn_node_value_to_string_out);
  ntype.expand_in_mf_network = fn_node_value_to_string_expand_in_mf_network;
  nodeRegisterType(&ntype);
}
