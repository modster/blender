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

#include "UI_interface.h"
#include "UI_resources.h"

#include "bmesh.h"
#include "bmesh_tools.h"

#include "node_geometry_util.hh"

static bNodeSocketTemplate geo_node_collapse_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_FLOAT, N_("Factor"), 1.0f, 0.0f, 0.0f, 0.0f, 0, 1.0f, PROP_FACTOR},
    {SOCK_STRING, N_("Selection")},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_collapse_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

static void geo_node_collapse_layout(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
{
  uiLayoutSetPropSep(layout, true);
  uiLayoutSetPropDecorate(layout, false);
  uiItemR(layout, ptr, "symmetry_axis", 0, nullptr, ICON_NONE);
}

static void geo_node_collapse_init(bNodeTree *UNUSED(tree), bNode *node)
{
  NodeGeometryCollapse *node_storage = (NodeGeometryCollapse *)MEM_callocN(
      sizeof(NodeGeometryCollapse), __func__);

  node->storage = node_storage;
  node_storage->symmetry_axis = GEO_NODE_COLLAPSE_SYMMETRY_AXIS_NONE;
}

namespace blender::nodes {

static Mesh *collapse_mesh(const float factor,
                           const VArray<float> &selection,
                           const bool triangulate,
                           const int symmetry_axis,
                           const Mesh *mesh)
{
  const BMeshCreateParams bmesh_create_params = {0};
  const BMeshFromMeshParams bmesh_from_mesh_params = {
      true, 0, 0, 0, {CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX}};
  BMesh *bm = BKE_mesh_to_bmesh_ex(mesh, &bmesh_create_params, &bmesh_from_mesh_params);

  const float symmetry_eps = 0.00002f;
  Array<float> mask(selection.size());
  selection.materialize(mask);

  BM_mesh_decimate_collapse(
      bm, factor, mask.data(), 1.0f, triangulate, symmetry_axis, symmetry_eps);
  Mesh *result = BKE_mesh_from_bmesh_for_eval_nomain(bm, NULL, mesh);
  BM_mesh_free(bm);

  result->runtime.cd_dirty_vert |= CD_MASK_NORMAL;
  return result;
}

static void geo_node_collapse_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");
  const float factor = params.extract_input<float>("Factor");

  if (factor < 1.0f && geometry_set.has_mesh()) {
    MeshComponent &mesh_component = geometry_set.get_component_for_write<MeshComponent>();

    const float default_factor = 1.0f;
    GVArray_Typed<float> selection_attribute = params.get_input_attribute<float>(
        "Selection", mesh_component, ATTR_DOMAIN_POINT, default_factor);
    // VArray<float> selection(selection_attribute.to);
    const Mesh *input_mesh = mesh_component.get_for_read();

    const bNode &node = params.node();
    const NodeGeometryCollapse &node_storage = *(NodeGeometryCollapse *)node.storage;
    Mesh *result = collapse_mesh(
        factor, selection_attribute, false, node_storage.symmetry_axis, input_mesh);
    geometry_set.replace_mesh(result);
  }

  params.set_output("Geometry", std::move(geometry_set));
}
}  // namespace blender::nodes

void register_node_type_geo_collapse()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_COLLAPSE, "Collapse", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, geo_node_collapse_in, geo_node_collapse_out);
  node_type_storage(
      &ntype, "NodeGeometryCollapse", node_free_standard_storage, node_copy_standard_storage);
  node_type_init(&ntype, geo_node_collapse_init);
  ntype.geometry_node_execute = blender::nodes::geo_node_collapse_exec;
  ntype.draw_buttons = geo_node_collapse_layout;
  ntype.width = 180;
  nodeRegisterType(&ntype);
}
