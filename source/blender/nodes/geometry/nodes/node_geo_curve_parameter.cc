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

static bNodeSocketTemplate geo_node_curve_parameter_out[] = {
    {SOCK_FLOAT, N_("Parameter")},
    {-1, ""},
};

namespace blender::nodes {

static void geo_node_curve_parameter_exec(GeoNodeExecParams params)
{
  FieldPtr curve_parameter_field = new bke::CurveParameterField();
  params.set_output("Parameter", bke::FieldRef<float>(std::move(curve_parameter_field)));
}

}  // namespace blender::nodes

void register_node_type_geo_curve_parameter()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_CURVE_PARAMETER, "Curve Parameter", NODE_CLASS_INPUT, 0);
  node_type_socket_templates(&ntype, nullptr, geo_node_curve_parameter_out);
  ntype.geometry_node_execute = blender::nodes::geo_node_curve_parameter_exec;
  nodeRegisterType(&ntype);
}
