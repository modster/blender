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

#include "node_geometry_util.hh"

static bNodeSocketTemplate geo_node_string_join_in[] = {
    {SOCK_STRING, N_("Delimiter")},
    {SOCK_STRING, N_("Strings"), 0.0f, 0.0f, 0.0f, 1.0f, -1.0f, 1.0f, PROP_NONE, SOCK_MULTI_INPUT},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_string_join_out[] = {
    {SOCK_STRING, N_("String")},
    {-1, ""},
};

namespace blender::nodes {

static void geo_node_string_join_exec(GeoNodeExecParams params)
{
  Vector<std::string> strings = params.extract_multi_input<std::string>("Strings");
  std::string delim = params.extract_input<std::string>("Delimiter");

  std::string output;
  int i = 0;
  for (std::string str : strings) {
    i++;
    output += str;
    if (i < strings.size()) {
      output += delim;
    }
  }
  params.set_output("String", output);
}

}  // namespace blender::nodes

void register_node_type_geo_string_join()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_STRING_JOIN, "String Join", NODE_CLASS_CONVERTOR, 0);
  node_type_socket_templates(&ntype, geo_node_string_join_in, geo_node_string_join_out);
  ntype.geometry_node_execute = blender::nodes::geo_node_string_join_exec;
  nodeRegisterType(&ntype);
}
