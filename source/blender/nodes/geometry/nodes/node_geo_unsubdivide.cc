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
    {SOCK_STRING, N_("Selection")},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_unsubdivide_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

namespace blender::nodes {

static Mesh *unsubdivide_mesh(const int iterations, const Array<bool> &selection, const Mesh *mesh)
{
  const BMeshCreateParams bmesh_create_params = {0};
  const BMeshFromMeshParams bmesh_from_mesh_params = {
      true, 0, 0, 0, {CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX}};
  BMesh *bm = BKE_mesh_to_bmesh_ex(mesh, &bmesh_create_params, &bmesh_from_mesh_params);
  BM_tag_vertices(bm, selection.data());
  BM_mesh_decimate_unsubdivide_ex(bm, iterations, true);

  Mesh *result = BKE_mesh_from_bmesh_for_eval_nomain(bm, NULL, mesh);
  BM_mesh_free(bm);
  result->runtime.cd_dirty_vert |= CD_MASK_NORMAL;

  return result;
}

static void geo_node_unsubdivide_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");
  const int iterations = params.extract_input<int>("Iterations");
  if (iterations > 0 && geometry_set.has_mesh()) {
    const MeshComponent &mesh_component = geometry_set.get_component_for_write<MeshComponent>();
    const Mesh *input_mesh = mesh_component.get_for_read();

    const bool default_selection = true;
    GVArray_Typed<bool> selection_attribute = params.get_input_attribute<bool>(
        "Selection", mesh_component, ATTR_DOMAIN_POINT, default_selection);
    VArray_Span<bool> selection{selection_attribute};

    Mesh *result = unsubdivide_mesh(iterations, selection, input_mesh);
    if (result != input_mesh) {
      geometry_set.replace_mesh(result);
    }
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
  ntype.width = 165;
  nodeRegisterType(&ntype);
}
