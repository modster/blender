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

namespace blender::nodes::node_geo_collapse {
static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>(N_("Mesh")).supported_type(GEO_COMPONENT_TYPE_MESH);
  b.add_input<decl::Float>(N_("Factor"))
      .default_value(1.0f)
      .min(0.0f)
      .max(1.0f)
      .subtype(PROP_FACTOR);
  b.add_input<decl::Bool>(N_("Selection")).default_value(true).supports_field().hide_value();
  b.add_output<decl::Geometry>(N_("Mesh"));
}

static void node_layout(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
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

static Mesh *collapse_mesh(const float factor,
                           const IndexMask selection,
                           const bool triangulate,
                           const int symmetry_axis,
                           const Mesh *mesh)
{
  const BMeshCreateParams bmesh_create_params = {0};
  const BMeshFromMeshParams bmesh_from_mesh_params = {
      true, 0, 0, 0, {CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX}};
  BMesh *bm = BKE_mesh_to_bmesh_ex(mesh, &bmesh_create_params, &bmesh_from_mesh_params);

  const float symmetry_eps = 0.00002f;

  /* Selection (bool) is converted to float array because BM_mesh_decimate_collapse takes it this
   * way. While from the description one could think that BM_mesh_decimate_collapse uses the actual
   * weight, it just uses it as mask. */
  Array<float> weights(mesh->totvert);
  for (int i_vert : selection) {
    weights[i_vert] = 1.0f;
  }

  BM_mesh_decimate_collapse(
      bm, factor, weights.begin(), 1.0f, triangulate, symmetry_axis, symmetry_eps);
  Mesh *result = BKE_mesh_from_bmesh_for_eval_nomain(bm, NULL, mesh);
  BM_mesh_free(bm);

  result->runtime.cd_dirty_vert |= CD_MASK_NORMAL;
  return result;
}

static void node_geo_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Mesh");
  const float factor = params.extract_input<float>("Factor");

  MeshComponent &mesh_component = geometry_set.get_component_for_write<MeshComponent>();

  Field<bool> selection_field = params.extract_input<Field<bool>>("Selection");
  const int domain_size = mesh_component.attribute_domain_size(ATTR_DOMAIN_POINT);
  GeometryComponentFieldContext context{mesh_component, ATTR_DOMAIN_POINT};
  FieldEvaluator evaluator{context, domain_size};
  evaluator.add(selection_field);
  evaluator.evaluate();

  geometry_set.modify_geometry_sets([&](GeometrySet &geometry_set) {
    if (geometry_set.has_mesh()) {
      const Mesh *input_mesh = mesh_component.get_for_read();

      const bNode &node = params.node();
      const NodeGeometryCollapse &node_storage = *(NodeGeometryCollapse *)node.storage;
      Mesh *result = collapse_mesh(factor,
                                   evaluator.get_evaluated_as_mask(0),
                                   false,
                                   node_storage.symmetry_axis,
                                   input_mesh);
      geometry_set.replace_mesh(result);
    }
  });

  params.set_output("Mesh", std::move(geometry_set));
}
}  // namespace blender::nodes::node_geo_collapse

void register_node_type_geo_collapse()
{
  namespace file_ns = blender::nodes::node_geo_collapse;

  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_COLLAPSE, "Collapse", NODE_CLASS_GEOMETRY);
  ntype.declare = file_ns::node_declare;
  node_type_storage(
      &ntype, "NodeGeometryCollapse", node_free_standard_storage, node_copy_standard_storage);
  node_type_init(&ntype, file_ns::geo_node_collapse_init);
  ntype.geometry_node_execute = file_ns::node_geo_exec;
  ntype.draw_buttons = file_ns::node_layout;
  ntype.width = 180;
  nodeRegisterType(&ntype);
}
