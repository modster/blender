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

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"

#include "BKE_mesh.h"
#include "BKE_node.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "bmesh.h"
#include "bmesh_tools.h"

#include "node_geometry_util.hh"

static bNodeSocketTemplate geo_node_extrude_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_BOOLEAN, N_("Individual")},
    {SOCK_BOOLEAN, N_("Selection"), 1, 0, 0, 0, 0, 1, PROP_NONE, SOCK_HIDE_VALUE},
    {SOCK_FLOAT, N_("Distance"), 0.0f, 0, 0, 0, FLT_MIN, FLT_MAX, PROP_DISTANCE},
    {SOCK_FLOAT, N_("Inset"), 0.0f, 0, 0, 0, FLT_MIN, FLT_MAX, PROP_DISTANCE},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_extrude_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_BOOLEAN, N_("Top Face"), 0, 0, 0, 0, 0, 0, PROP_NONE, SOCK_IS_ATTRIBUTE_OUTPUT},
    {SOCK_BOOLEAN, N_("Side Face"), 0, 0, 0, 0, 0, 0, PROP_NONE, SOCK_IS_ATTRIBUTE_OUTPUT},
    {-1, ""},
};

using blender::Span;

static Mesh *extrude_mesh(const Mesh *mesh,
                          const Span<bool> selection,
                          const Span<float> distance,
                          const Span<float> inset,
                          const bool inset_individual_faces,
                          bool **selection_top_faces_out,
                          bool **selection_all_faces_out)
{
  const BMeshCreateParams bmesh_create_params = {true};
  const BMeshFromMeshParams bmesh_from_mesh_params = {
      true, 0, 0, 0, {CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX}};

  BMesh *bm = BKE_mesh_to_bmesh_ex(mesh, &bmesh_create_params, &bmesh_from_mesh_params);

  BM_select_faces(bm, selection.data());
  BMOperator op;
  if (inset_individual_faces) {
    BMO_op_initf(bm,
                 &op,
                 0,
                 "inset_individual faces=%hf use_even_offset=%b thickness=%f depth=%f "
                 "thickness_array=%p depth_array=%p use_attributes=%b",
                 BM_ELEM_SELECT,
                 true,
                 inset[0],
                 distance[0],
                 inset.data(),
                 distance.data(),
                 true);
  }
  else {
    BMO_op_initf(bm,
                 &op,
                 0,
                 "inset_region faces=%hf use_boundary=%b use_even_offset=%b thickness=%f depth=%f "
                 "thickness_array=%p depth_array=%p use_attributes=%b",
                 BM_ELEM_SELECT,
                 true,
                 true,
                 inset[0],
                 distance[0],
                 inset.data(),
                 distance.data(),
                 true);
  }
  BM_tag_new_faces(bm, &op);
  BMO_op_exec(bm, &op);
  BMO_op_finish(bm, &op);

  CustomData_MeshMasks cd_mask_extra = {CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX};

  Mesh *result = BKE_mesh_from_bmesh_for_eval_nomain(bm, &cd_mask_extra, mesh);
  BM_get_selected_faces(bm, selection_top_faces_out);
  BM_get_tagged_faces(bm, selection_all_faces_out);

  BM_mesh_free(bm);

  BKE_mesh_normals_tag_dirty(result);

  return result;
}

namespace blender::nodes {
static void geo_node_extrude_exec(GeoNodeExecParams params)
{
  const bNode &node = params.node();
  const bNodeTree &ntree = params.ntree();
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");

  geometry_set = geometry_set_realize_instances(geometry_set);

  Array<bool> top_faces;
  Array<bool> side_faces;

  MeshComponent &mesh_component = geometry_set.get_component_for_write<MeshComponent>();
  if (mesh_component.has_mesh()) {
    const Mesh *input_mesh = mesh_component.get_for_read();

    Array<bool> selection_array = params.extract_input<Array<bool>>("Selection");
    fn::GVArray_For_RepeatedGSpan selection_repeated{input_mesh->totpoly,
                                                     selection_array.as_span()};
    fn::GVArray_Span<bool> selection{selection_repeated};

    AttributeDomain attribute_domain = ATTR_DOMAIN_POINT;
    const bool inset_individual_faces = params.extract_input<bool>("Individual");

    if (inset_individual_faces) {
      attribute_domain = ATTR_DOMAIN_FACE;
    }
    const int domain_size = mesh_component.attribute_domain_size(attribute_domain);

    Array<float> distances_array = params.extract_input<Array<float>>("Distance");
    fn::GVArray_For_RepeatedGSpan distances_repeated{domain_size, distances_array.as_span()};
    fn::GVArray_Span<float> distance{distances_repeated};

    Array<float> insets_array = params.extract_input<Array<float>>("Inset");
    fn::GVArray_For_RepeatedGSpan insets_repeated{domain_size, insets_array.as_span()};
    fn::GVArray_Span<float> inset{insets_repeated};

    bool *selection_top_faces_out = nullptr;
    bool *selection_all_faces_out = nullptr;

    Mesh *result = extrude_mesh(input_mesh,
                                selection,
                                distance,
                                inset,
                                inset_individual_faces,
                                &selection_top_faces_out,
                                &selection_all_faces_out);

    geometry_set.replace_mesh(result);

    MeshComponent &result_mesh_component = geometry_set.get_component_for_write<MeshComponent>();

    for (const int i : IndexRange(result->totpoly)) {
      if (selection_top_faces_out[i]) {
        selection_all_faces_out[i] = false;
      }
    }

    top_faces = Span(selection_top_faces_out, result->totpoly);
    side_faces = Span(selection_all_faces_out, result->totpoly);

    if (should_add_output_attribute(node, "Top Face")) {
      std::string attribute_name = get_local_attribute_name(ntree.id.name, node.name, "Top Face");
      fn::GVArray_For_Span varray{top_faces.as_span()};
      result_mesh_component.attribute_try_create(
          attribute_name, ATTR_DOMAIN_FACE, CD_PROP_BOOL, AttributeInitVArray{&varray});
    }
    if (should_add_output_attribute(node, "Side Face")) {
      std::string attribute_name = get_local_attribute_name(ntree.id.name, node.name, "Side Face");
      fn::GVArray_For_Span varray{side_faces.as_span()};
      result_mesh_component.attribute_try_create(
          attribute_name, ATTR_DOMAIN_FACE, CD_PROP_BOOL, AttributeInitVArray{&varray});
    }

    MEM_freeN(selection_top_faces_out);
    MEM_freeN(selection_all_faces_out);
  }

  params.set_output("Geometry", std::move(geometry_set));
  params.set_output("Top Face", std::move(top_faces));
  params.set_output("Side Face", std::move(side_faces));
}
}  // namespace blender::nodes

void register_node_type_geo_extrude()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_EXTRUDE, "Extrude", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, geo_node_extrude_in, geo_node_extrude_out);
  ntype.geometry_node_execute = blender::nodes::geo_node_extrude_exec;
  nodeRegisterType(&ntype);
}
