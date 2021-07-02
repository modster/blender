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
#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"
#include "BKE_mesh.h"
#include "bmesh.h"
#include "bmesh_tools.h"
#include "node_geometry_util.hh"
#include "math.h"

static bNodeSocketTemplate geo_node_dissolve_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_FLOAT, N_("Angle"), M_PI * 0.25, 0.0f, 0.0f, 1.0f, 0.0f, M_PI, PROP_ANGLE},
    {SOCK_BOOLEAN, N_("All Boundaries"), false},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_dissolve_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

namespace blender::nodes{

static Mesh *dissolveMesh(const float angle, const bool all_boundaries, const int delimiter, Mesh *mesh)
{
  BMesh *bm;
  BMeshCreateParams bmesh_create_params = {0};
  BMeshFromMeshParams bmesh_from_mesh_params = {true ,0,0,0,{CD_MASK_ORIGINDEX,CD_MASK_ORIGINDEX,CD_MASK_ORIGINDEX}};
  bm = BKE_mesh_to_bmesh_ex(mesh, &bmesh_create_params, &bmesh_from_mesh_params);

  BM_mesh_decimate_dissolve(bm, angle, all_boundaries, (BMO_Delimit)delimiter);

  Mesh *result = BKE_mesh_from_bmesh_for_eval_nomain(bm, NULL, mesh);
  BM_mesh_free(bm);
  result->runtime.cd_dirty_vert |= CD_MASK_NORMAL;
  return result;
}

static void geo_node_dissolve_init(bNodeTree *UNUSED(tree), bNode *node)
{
  NodeGeometryDissolve *node_storage = (NodeGeometryDissolve *)MEM_callocN(
      sizeof(NodeGeometryDissolve), __func__);

  node->storage = node_storage;
  node_storage->delimiter = GEO_NODE_DISSOLVE_DELIMITTER_NORMAL;
}

static void geo_node_dissolve_layout(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
{
  uiLayoutSetPropSep(layout, true);
  uiLayoutSetPropDecorate(layout, false);
  uiItemR(layout, ptr, "delimiter", 0, nullptr, ICON_NONE);
}

static void geo_node_dissolve_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");
  float angle = params.extract_input<float>("Angle");

  if(geometry_set.has_mesh()){
    MeshComponent &mesh_component = geometry_set.get_component_for_write<MeshComponent>();

    Mesh *input_mesh = mesh_component.get_for_write();

    if (input_mesh->totvert <= 3){
      params.error_message_add(NodeWarningType::Error,
                               TIP_("Node requires mesh with more than 3 input faces"));
    }

    const bool all_boundaries = params.extract_input<bool>("All Boundaries");
    const bNode &node = params.node();
    NodeGeometryDissolve &node_storage = *(NodeGeometryDissolve *)node.storage;
    Mesh *result = dissolveMesh(angle, all_boundaries , node_storage.delimiter, input_mesh);
    geometry_set.replace_mesh(result);
  }

  params.set_output("Geometry", std::move(geometry_set));
}
}  // namespace blender::nodes

void register_node_type_geo_dissolve()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_DISSOLVE, "Dissolve", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, geo_node_dissolve_in, geo_node_dissolve_out);
  node_type_storage(
      &ntype, "NodeGeometryDissolve", node_free_standard_storage, node_copy_standard_storage);
  node_type_init(&ntype, blender::nodes::geo_node_dissolve_init);
  ntype.geometry_node_execute = blender::nodes::geo_node_dissolve_exec;
  ntype.draw_buttons = blender::nodes::geo_node_dissolve_layout;
  nodeRegisterType(&ntype);
}
