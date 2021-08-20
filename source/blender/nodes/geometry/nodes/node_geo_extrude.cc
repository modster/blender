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
    {SOCK_FLOAT, N_("Distance"), 0.0f, 0, 0, 0, FLT_MIN, FLT_MAX, PROP_DISTANCE, SOCK_FIELD},
    {SOCK_FLOAT, N_("Inset"), 0.0f, 0, 0, 0, FLT_MIN, FLT_MAX, PROP_DISTANCE, SOCK_FIELD},
    {SOCK_BOOLEAN, N_("Individual")},
    {SOCK_BOOLEAN, N_("Selection"), 1, 0, 0, 0, 0, 0, PROP_NONE, SOCK_HIDE_VALUE | SOCK_FIELD},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_extrude_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_BOOLEAN, N_("Top Faces")},
    {SOCK_BOOLEAN, N_("Side Faces")},
    {-1, ""},
};

namespace blender::nodes {

static Mesh *extrude_mesh(const Mesh *mesh,
                          const Span<bool> selection,
                          const Span<float> distance,
                          const Span<float> inset,
                          const bool inset_individual_faces,
                          AnonymousCustomDataLayerID *top_faces_id,
                          AnonymousCustomDataLayerID *side_faces_id)
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
                 inset.first(),
                 distance.first(),
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
                 inset.first(),
                 distance.first(),
                 inset.data(),
                 distance.data(),
                 true);
  }
  BMO_op_exec(bm, &op);
  BM_tag_new_faces(bm, &op);

  CustomData_MeshMasks cd_mask_extra = {CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX};
  Mesh *result = BKE_mesh_from_bmesh_for_eval_nomain(bm, &cd_mask_extra, mesh);

  MeshComponent component;
  component.replace(result, GeometryOwnershipType::Editable);

  if (side_faces_id) {
    OutputAttribute_Typed<bool> attribute =
        component.attribute_try_get_anonymous_for_output_only<bool>(*side_faces_id,
                                                                    ATTR_DOMAIN_FACE);
    BM_get_tagged_faces(bm, attribute.as_span().data());
    attribute.save();
  }
  if (top_faces_id) {
    OutputAttribute_Typed<bool> attribute =
        component.attribute_try_get_anonymous_for_output_only<bool>(*top_faces_id,
                                                                    ATTR_DOMAIN_FACE);
    BM_get_selected_faces(bm, attribute.as_span().data());
    attribute.save();
  }

  BMO_op_finish(bm, &op);
  BM_mesh_free(bm);

  BKE_mesh_normals_tag_dirty(result);

  return result;
}

static void geo_node_extrude_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");

  geometry_set = bke::geometry_set_realize_instances(geometry_set);

  MeshComponent &mesh_component = geometry_set.get_component_for_write<MeshComponent>();
  if (mesh_component.has_mesh() && mesh_component.get_for_read()->totpoly > 0) {
    const Mesh *input_mesh = mesh_component.get_for_read();

    bke::FieldRef<bool> field = params.get_input_field<bool>("Selection");
    bke::FieldInputs field_inputs = field->prepare_inputs();
    Vector<std::unique_ptr<bke::FieldInputValue>> field_input_values;
    prepare_field_inputs(field_inputs, mesh_component, ATTR_DOMAIN_FACE, field_input_values);
    bke::FieldOutput field_output = field->evaluate(IndexRange(input_mesh->totpoly), field_inputs);
    GVArray_Typed<bool> selection_results{field_output.varray_ref()};
    VArray_Span<bool> selection{selection_results};

    const bool inset_individual_faces = params.extract_input<bool>("Individual");
    const AttributeDomain domain = inset_individual_faces ? ATTR_DOMAIN_FACE : ATTR_DOMAIN_POINT;

    bke::FieldRef<float> distance_field = params.get_input_field<float>("Distance");
    bke::FieldInputs distance_field_inputs = distance_field->prepare_inputs();
    Vector<std::unique_ptr<bke::FieldInputValue>> distance_field_input_values;
    prepare_field_inputs(
        distance_field_inputs, mesh_component, domain, distance_field_input_values);
    bke::FieldOutput distance_field_output = distance_field->evaluate(
        IndexRange(mesh_component.attribute_domain_size(domain)), distance_field_inputs);
    GVArray_Typed<float> distance_results{distance_field_output.varray_ref()};
    VArray_Span<float> distance{distance_results};

    bke::FieldRef<float> inset_field = params.get_input_field<float>("Inset");
    bke::FieldInputs inset_field_inputs = inset_field->prepare_inputs();
    Vector<std::unique_ptr<bke::FieldInputValue>> inset_field_input_values;
    prepare_field_inputs(inset_field_inputs, mesh_component, domain, inset_field_input_values);
    bke::FieldOutput inset_field_output = inset_field->evaluate(
        IndexRange(mesh_component.attribute_domain_size(domain)), inset_field_inputs);
    GVArray_Typed<float> inset_results{inset_field_output.varray_ref()};
    VArray_Span<float> inset{inset_results};

    AnonymousCustomDataLayerID *top_faces_id = params.output_is_required("Top Faces") ?
                                                   CustomData_anonymous_id_new("Top Faces") :
                                                   nullptr;
    AnonymousCustomDataLayerID *side_faces_id = params.output_is_required("Side Faces") ?
                                                    CustomData_anonymous_id_new("Side Faces") :
                                                    nullptr;

    Mesh *result = extrude_mesh(input_mesh,
                                selection,
                                distance,
                                inset,
                                inset_individual_faces,
                                top_faces_id,
                                side_faces_id);
    geometry_set.replace_mesh(result);

    if (top_faces_id) {
      params.set_output("Top Faces",
                        bke::FieldRef<bool>(new bke::AnonymousAttributeField(
                            *top_faces_id, CPPType::get<bool>())));
    }
    if (side_faces_id) {
      params.set_output("Side Faces",
                        bke::FieldRef<bool>(new bke::AnonymousAttributeField(
                            *side_faces_id, CPPType::get<bool>())));
    }
  }

  params.set_output("Geometry", std::move(geometry_set));
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
