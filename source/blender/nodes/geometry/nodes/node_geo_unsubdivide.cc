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

namespace blender::nodes::node_geo_unsubdivide {

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>(N_("Mesh")).supported_type(GEO_COMPONENT_TYPE_MESH);
  b.add_input<decl::Int>(N_("Iterations")).default_value(1).min(0).max(10);
  b.add_input<decl::Bool>(N_("Selection")).default_value(true).supports_field().hide_value();
  b.add_output<decl::Geometry>(N_("Mesh"));
}

static Mesh *unsubdivide_mesh(const int iterations, const IndexMask selection, const Mesh *mesh)
{
  const BMeshCreateParams bmesh_create_params = {0};
  const BMeshFromMeshParams bmesh_from_mesh_params = {
      true, 0, 0, 0, {CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX}};
  BMesh *bm = BKE_mesh_to_bmesh_ex(mesh, &bmesh_create_params, &bmesh_from_mesh_params);

  BM_mesh_elem_table_ensure(bm, BM_VERT);
  for (int i_point : selection) {
    BM_elem_flag_set(BM_vert_at_index(bm, i_point), BM_ELEM_TAG, true);
  }

  BM_mesh_decimate_unsubdivide_ex(bm, iterations, true);

  Mesh *result = BKE_mesh_from_bmesh_for_eval_nomain(bm, NULL, mesh);
  BM_mesh_free(bm);
  result->runtime.cd_dirty_vert |= CD_MASK_NORMAL;

  return result;
}

static void node_geo_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Mesh");
  const int iterations = params.extract_input<int>("Iterations");
  if (iterations > 0 && geometry_set.has_mesh()) {
    const MeshComponent &mesh_component = geometry_set.get_component_for_write<MeshComponent>();
    const Mesh *input_mesh = mesh_component.get_for_read();

    Field<bool> selection_field = params.extract_input<Field<bool>>("Selection");
    const int domain_size = mesh_component.attribute_domain_size(ATTR_DOMAIN_POINT);
    GeometryComponentFieldContext context{mesh_component, ATTR_DOMAIN_POINT};
    FieldEvaluator evaluator{context, domain_size};
    evaluator.add(selection_field);
    evaluator.evaluate();
    const IndexMask selection = evaluator.get_evaluated_as_mask(0);

    geometry_set.modify_geometry_sets([&](GeometrySet &geometry_set) {
      Mesh *result = unsubdivide_mesh(iterations, selection, input_mesh);
      if (result != input_mesh) {
        geometry_set.replace_mesh(result);
      }
    });
  }
  params.set_output("Mesh", std::move(geometry_set));
}
}  // namespace blender::nodes::node_geo_unsubdivide

void register_node_type_geo_unsubdivide()
{
  namespace file_ns = blender::nodes::node_geo_unsubdivide;

  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_UNSUBDIVIDE, "Unsubdivide", NODE_CLASS_GEOMETRY);
  ntype.declare = file_ns::node_declare;
  ntype.geometry_node_execute = file_ns::node_geo_exec;
  nodeRegisterType(&ntype);
}
