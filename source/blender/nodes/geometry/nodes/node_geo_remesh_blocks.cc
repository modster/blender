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

#include "BKE_mesh.h"
#include "BKE_mesh_remesh_blocks.h"

#include "UI_interface.h"

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"
#include "DNA_modifier_types.h"

#include "node_geometry_util.hh"

static bNodeSocketTemplate geo_node_remesh_blocks_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_INT, N_("Depth"), 4, 0, 0, 0, 2, 64},
    {SOCK_FLOAT, N_("Scale"), 0.9f, 0, 0, 0, 0.0f, 0.99f},
    {SOCK_FLOAT, N_("Threshold"), 1.0f, 0, 0, 0, 0.01f, FLT_MAX},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_remesh_blocks_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

namespace blender::nodes {
static void geo_node_remesh_blocks_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");
  const char flag = MOD_REMESH_FLOOD_FILL;
  const char mode = 0;
  const int hermite_num = 1;
  const int depth = params.extract_input<int>("Depth");
  const float scale = params.extract_input<float>("Scale");
  const float threshold = params.extract_input<float>("Threshold");

  if (geometry_set.has_mesh()) {
    Mesh *input_mesh = geometry_set.get_mesh_for_write();

    Mesh *output_mesh = BKE_mesh_remesh_blocks_to_mesh_nomain(
        input_mesh, flag, mode, threshold, hermite_num, scale, depth);
    for(int i = 0; i < output_mesh->totpoly; i++){
      printf("flag: %i\n",output_mesh->mpoly[i].flag);
    }
    BKE_mesh_copy_parameters_for_eval(output_mesh, input_mesh);
    BKE_mesh_calc_edges(input_mesh, true, false);
    output_mesh->runtime.cd_dirty_vert |= CD_MASK_NORMAL;

    geometry_set.replace_mesh(output_mesh);
  }
  params.set_output("Geometry", std::move(geometry_set));
}
}  // namespace blender::nodes

void register_node_type_geo_remesh_blocks()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_REMESH_BLOCKS, "Remesh Blocks", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, geo_node_remesh_blocks_in, geo_node_remesh_blocks_out);
  ntype.geometry_node_execute = blender::nodes::geo_node_remesh_blocks_exec;
  nodeRegisterType(&ntype);
}
