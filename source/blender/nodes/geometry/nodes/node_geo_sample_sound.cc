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

namespace blender::nodes {

static void geo_node_sample_sound_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Float>(N_("Frame")).supports_field();
  b.add_input<decl::Float>(N_("Min Frequency")).supports_field().default_value(0.0f);
  b.add_input<decl::Float>(N_("Max Frequency")).supports_field().default_value(20000.0f);
  b.add_output<decl::Float>(N_("Volume")).dependent_field();
}

static void geo_node_sample_sound_exec(GeoNodeExecParams params)
{
  params.set_output("Volume", 0.0f);
}

}  // namespace blender::nodes

void register_node_type_geo_sample_sound()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_SAMPLE_SOUND, "Sample Sound", NODE_CLASS_TEXTURE, 0);
  node_type_size(&ntype, 200, 40, 1000);
  ntype.declare = blender::nodes::geo_node_sample_sound_declare;
  ntype.geometry_node_execute = blender::nodes::geo_node_sample_sound_exec;
  nodeRegisterType(&ntype);
}
