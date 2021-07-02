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

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"

#include "bmesh.h"
#include "bmesh_tools.h"
#include "node_geometry_util.hh"

static bNodeSocketTemplate geo_node_unsubdivide_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_INT, N_("Iterations"), 1, 0, 0, 0, 0, 10},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_unsubdivide_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

namespace blender::nodes {

static Mesh *unsubdivideMesh(const uint iterations, Mesh *mesh)
{
  BMesh *bm;

  if (iterations == 0) {
    return mesh;
  }

  BMeshCreateParams bmesh_create_params = {0};
  BMeshFromMeshParams bmesh_from_mesh_params = {
      true, 0, 0, 0, {CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX}};
  bm = BKE_mesh_to_bmesh_ex(mesh, &bmesh_create_params, &bmesh_from_mesh_params);

  BM_mesh_decimate_unsubdivide(bm, iterations);

  Mesh *result = BKE_mesh_from_bmesh_for_eval_nomain(bm, NULL, mesh);
  BM_mesh_free(bm);
  result->runtime.cd_dirty_vert |= CD_MASK_NORMAL;

  return result;
}

static void geo_node_unsubdivide_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");
  int iterations = params.extract_input<int>("Iterations");
  if (geometry_set.has_mesh()) {
    Mesh *input_mesh = geometry_set.get_mesh_for_write();
    Mesh *result = unsubdivideMesh(iterations, input_mesh);
    geometry_set.replace_mesh(result);
  }
  params.set_output("Geometry", std::move(geometry_set));
}
}  // namespace blender::nodes

void register_node_type_geo_unsubdivide()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_UNSUBDIVIDE, "Unsubdivide", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, geo_node_unsubdivide_in, geo_node_unsubdivide_out);
  ntype.geometry_node_execute = blender::nodes::geo_node_unsubdivide_exec;
  nodeRegisterType(&ntype);
}
