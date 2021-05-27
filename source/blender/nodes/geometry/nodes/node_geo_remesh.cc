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
#include "BKE_mesh_remesh_voxel.h"

#include "node_geometry_util.hh"

static bNodeSocketTemplate geo_node_remesh_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_FLOAT, N_("Voxel Size"), 1.0f, 0, 0, 0, 0.01f, FLT_MAX},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_remesh_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

namespace blender::nodes {
static void geo_node_remesh_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");
  float voxel_size = params.extract_input<float>("Voxel Size");
  if(geometry_set.has_mesh()){
    Mesh *input_mesh = geometry_set.get_mesh_for_write();
    Mesh *output_mesh = BKE_mesh_remesh_voxel_to_mesh_nomain(input_mesh, voxel_size, 0.5f, 0.0f);
    geometry_set.replace_mesh(output_mesh);
  }
  params.set_output("Geometry", std::move(geometry_set));
}
}  // namespace blender::nodes

void register_node_type_geo_remesh()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_REMESH, "remesh", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, geo_node_remesh_in, geo_node_remesh_out);
  ntype.geometry_node_execute = blender::nodes::geo_node_remesh_exec;
  nodeRegisterType(&ntype);
}
