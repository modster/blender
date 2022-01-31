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
#include "math.h"
#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_dissolve {

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>(N_("Mesh")).supported_type(GEO_COMPONENT_TYPE_MESH);
  b.add_input<decl::Float>(N_("Angle"))
      .default_value(0.0f)
      .min(0.0f)
      .max(M_PI)
      .subtype(PROP_ANGLE);
  b.add_input<decl::Bool>(N_("All Boundaries")).default_value(false);
  b.add_input<decl::Bool>(N_("Selection")).default_value(true).supports_field().hide_value();
  b.add_output<decl::Geometry>(N_("Mesh"));
}

static void node_layout(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
{
  uiLayoutSetPropSep(layout, true);
  uiLayoutSetPropDecorate(layout, false);
  uiItemR(layout, ptr, "selection_type", 0, nullptr, ICON_NONE);
}

static void geo_node_dissolve_init(bNodeTree *UNUSED(tree), bNode *node)
{
  NodeGeometryDissolve *node_storage = (NodeGeometryDissolve *)MEM_callocN(
      sizeof(NodeGeometryDissolve), __func__);

  node->storage = node_storage;
  node_storage->selection_type = GEO_NODE_DISSOLVE_DELIMITTER_UNSELECTED;
}

static Mesh *dissolve_mesh(const float angle,
                           const bool all_boundaries,
                           const int delimiter,
                           const IndexMask selection,
                           const Mesh *mesh)
{
  const BMeshCreateParams bmesh_create_params = {0};
  const BMeshFromMeshParams bmesh_from_mesh_params = {
      true, 0, 0, 0, {CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX}};
  BMesh *bm = BKE_mesh_to_bmesh_ex(mesh, &bmesh_create_params, &bmesh_from_mesh_params);
  if (delimiter & BMO_DELIM_FACE_SELECTION) {
    // BM_tag_faces(bm, selection.data());
    BM_mesh_elem_table_ensure(bm, BM_FACE);
    for (int i_face : selection) {
      BM_elem_flag_set(BM_face_at_index(bm, i_face), BM_ELEM_TAG, true);
    }
  }
  else {
    // BM_select_edges(bm, selection.data());
    for (int i_edge : selection) {
      BM_mesh_elem_table_ensure(bm, BM_EDGE);
      BM_elem_flag_set(BM_edge_at_index(bm, i_edge), BM_ELEM_SELECT, true);
    }
  }

  BM_mesh_decimate_dissolve(bm, angle, all_boundaries, (BMO_Delimit)delimiter);

  Mesh *result = BKE_mesh_from_bmesh_for_eval_nomain(bm, NULL, mesh);
  BM_mesh_free(bm);
  result->runtime.cd_dirty_vert |= CD_MASK_NORMAL;
  return result;
}

static void node_geo_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Mesh");
  const float angle = params.extract_input<float>("Angle");

  if (angle > 0.0f && geometry_set.has_mesh()) {
    const MeshComponent &mesh_component = geometry_set.get_component_for_write<MeshComponent>();
    const Mesh *input_mesh = mesh_component.get_for_read();

    const bool all_boundaries = params.extract_input<bool>("All Boundaries");
    const bNode &node = params.node();
    const NodeGeometryDissolve &node_storage = *(NodeGeometryDissolve *)node.storage;

    // bool default_selection = false;
    AttributeDomain selection_domain = ATTR_DOMAIN_FACE;
    BMO_Delimit delimiter = BMO_DELIM_FACE_SELECTION;

    if (node_storage.selection_type == GEO_NODE_DISSOLVE_DELIMITTER_UNSELECTED) {
      selection_domain = ATTR_DOMAIN_EDGE;
      delimiter = BMO_DELIM_EDGE_SELECTION_INVSE;
    }
    else if (node_storage.selection_type == GEO_NODE_DISSOLVE_DELIMITTER_LIMIT) {
      selection_domain = ATTR_DOMAIN_EDGE;
      delimiter = BMO_DELIM_EDGE_SELECTION;
      // default_selection = true;
    };

    Field<bool> selection_field = params.extract_input<Field<bool>>("Selection");
    const int domain_size = mesh_component.attribute_domain_size(selection_domain);
    GeometryComponentFieldContext context{mesh_component, selection_domain};
    FieldEvaluator evaluator{context, domain_size};
    evaluator.add(selection_field);
    evaluator.evaluate();
    const IndexMask selection = evaluator.get_evaluated_as_mask(0);

    geometry_set.modify_geometry_sets([&](GeometrySet &geometry_set) {
      Mesh *result = dissolve_mesh(angle, all_boundaries, delimiter, selection, input_mesh);
      geometry_set.replace_mesh(result);
    });
  }

  params.set_output("Mesh", std::move(geometry_set));
}
}  // namespace blender::nodes::node_geo_dissolve

void register_node_type_geo_dissolve()
{
  namespace file_ns = blender::nodes::node_geo_dissolve;

  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_DISSOLVE, "Dissolve", NODE_CLASS_GEOMETRY);
  ntype.declare = file_ns::node_declare;

  node_type_storage(
      &ntype, "NodeGeometryDissolve", node_free_standard_storage, node_copy_standard_storage);
  node_type_init(&ntype, file_ns::geo_node_dissolve_init);
  ntype.geometry_node_execute = file_ns::node_geo_exec;
  ntype.draw_buttons = file_ns::node_layout;
  ntype.width = 165;
  nodeRegisterType(&ntype);
}