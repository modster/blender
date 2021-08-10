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

static bNodeSocketTemplate geo_node_extrude_and_move_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_VECTOR, N_("Offset"), 0.0f, 0.0f, 1.0f, 0.0f, -FLT_MAX, FLT_MAX, PROP_TRANSLATION},
    {SOCK_BOOLEAN, N_("Selection"), 1, 0, 0, 0, 0, 0, PROP_NONE, SOCK_HIDE_VALUE | SOCK_FIELD},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_extrude_and_move_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_BOOLEAN, N_("Selection")},
    {-1, ""},
};

static void geo_node_extrude_and_move_layout(uiLayout *layout,
                                             bContext *UNUSED(C),
                                             PointerRNA *ptr)
{
  uiItemR(layout, ptr, "extrude_mode", 0, "", ICON_NONE);
}

static void geo_node_extrude_and_move_init(bNodeTree *UNUSED(ntree), bNode *node)
{
  node->custom1 = 0;
}

namespace blender::nodes {

static Mesh *extrude_vertices(const Mesh *mesh,
                              const Span<bool> selection,
                              const float3 offset,
                              AnonymousCustomDataLayerID *out_selection_id)
{
  const BMeshCreateParams bmesh_create_params = {true};
  const BMeshFromMeshParams bmesh_from_mesh_params = {
      true, 0, 0, 0, {CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX}};

  BMesh *bm = BKE_mesh_to_bmesh_ex(mesh, &bmesh_create_params, &bmesh_from_mesh_params);

  BM_select_vertices(bm, selection.data());

  BMOperator extrude_op;
  BMO_op_initf(bm, &extrude_op, 0, "extrude_vert_indiv verts=%hv", BM_ELEM_SELECT);
  BMO_op_exec(bm, &extrude_op);

  float mx[3] = {offset.x, offset.y, offset.z};
  BMOperator move_op;

  BM_mesh_elem_hflag_disable_all(bm, BM_VERT, BM_ELEM_SELECT, false);
  BMO_slot_buffer_hflag_enable(
      bm, extrude_op.slots_out, "verts.out", BM_VERT, BM_ELEM_SELECT, false);

  BMO_op_initf(bm, &move_op, 0, "translate vec=%v verts=%hv", mx, BM_ELEM_SELECT);

  BMO_op_exec(bm, &move_op);
  BMO_op_finish(bm, &move_op);
  BMO_op_finish(bm, &extrude_op);

  CustomData_MeshMasks cd_mask_extra = {CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX};
  Mesh *result = BKE_mesh_from_bmesh_for_eval_nomain(bm, &cd_mask_extra, mesh);

  MeshComponent component;
  component.replace(result, GeometryOwnershipType::Editable);

  if (out_selection_id) {
    OutputAttribute_Typed<bool> attribute =
        component.attribute_try_get_anonymous_for_output_only<bool>(*out_selection_id,
                                                                    ATTR_DOMAIN_POINT);
    BM_get_selected_vertices(bm, attribute.as_span().data());
    attribute.save();
  }

  BM_mesh_free(bm);

  BKE_mesh_normals_tag_dirty(result);

  return result;
}

static Mesh *extrude_edges(const Mesh *mesh,
                           const Span<bool> selection,
                           const float3 offset,
                           AnonymousCustomDataLayerID *out_selection_id)
{
  const BMeshCreateParams bmesh_create_params = {true};
  const BMeshFromMeshParams bmesh_from_mesh_params = {
      true, 0, 0, 0, {CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX}};

  BMesh *bm = BKE_mesh_to_bmesh_ex(mesh, &bmesh_create_params, &bmesh_from_mesh_params);

  BM_select_edges(bm, selection.data());

  BMOperator extrude_op;
  BMO_op_initf(bm,
               &extrude_op,
               0,
               "extrude_edge_only edges=%he use_select_history=%b",
               BM_ELEM_SELECT,
               true);
  BMO_op_exec(bm, &extrude_op);

  float o[3] = {offset.x, offset.y, offset.z};
  BMOperator move_op;

  BM_mesh_elem_hflag_disable_all(bm, BM_VERT, BM_ELEM_SELECT, false);
  BMO_slot_buffer_hflag_enable(
      bm, extrude_op.slots_out, "geom.out", BM_VERT, BM_ELEM_SELECT, false);

  BMO_op_initf(bm, &move_op, 0, "translate vec=%v verts=%hv", o, BM_ELEM_SELECT);

  BMO_op_exec(bm, &move_op);
  BMO_op_finish(bm, &move_op);
  BMO_op_finish(bm, &extrude_op);

  CustomData_MeshMasks cd_mask_extra = {CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX};
  Mesh *result = BKE_mesh_from_bmesh_for_eval_nomain(bm, &cd_mask_extra, mesh);

  MeshComponent component;
  component.replace(result, GeometryOwnershipType::Editable);

  if (out_selection_id) {
    OutputAttribute_Typed<bool> attribute =
        component.attribute_try_get_anonymous_for_output_only<bool>(*out_selection_id,
                                                                    ATTR_DOMAIN_POINT);
    BM_get_selected_vertices(bm, attribute.as_span().data());
    attribute.save();
  }

  BM_mesh_free(bm);

  BKE_mesh_normals_tag_dirty(result);

  return result;
}

static Mesh *extrude_faces(const Mesh *mesh,
                           const Span<bool> selection,
                           const float3 offset,
                           AnonymousCustomDataLayerID *out_selection_id)
{
  // TODO: - dont execute on a offset with length 0
  //       - Check why selection for edges and faces is wired.
  //       - dedublicate extrude functions
  //       - checkout hans lazy bmesh mechanism

  const BMeshCreateParams bmesh_create_params = {true};
  const BMeshFromMeshParams bmesh_from_mesh_params = {
      true, 0, 0, 0, {CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX}};

  BMesh *bm = BKE_mesh_to_bmesh_ex(mesh, &bmesh_create_params, &bmesh_from_mesh_params);

  BM_select_faces(bm, selection.data());

  BMOperator extrude_op;
  BMO_op_initf(bm, &extrude_op, 0, "extrude_face_region geom=%hf", BM_ELEM_SELECT);
  BMO_op_exec(bm, &extrude_op);

  float o[3] = {offset.x, offset.y, offset.z};
  BMOperator move_op;

  BM_mesh_elem_hflag_disable_all(bm, BM_VERT, BM_ELEM_SELECT, false);
  BMO_slot_buffer_hflag_enable(
      bm, extrude_op.slots_out, "geom.out", BM_VERT, BM_ELEM_SELECT, false);

  BMO_op_initf(bm, &move_op, 0, "translate vec=%v verts=%hv", o, BM_ELEM_SELECT);

  BMO_op_exec(bm, &move_op);
  BMO_op_finish(bm, &move_op);
  BMO_op_finish(bm, &extrude_op);

  CustomData_MeshMasks cd_mask_extra = {CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX};
  Mesh *result = BKE_mesh_from_bmesh_for_eval_nomain(bm, &cd_mask_extra, mesh);

  MeshComponent component;
  component.replace(result, GeometryOwnershipType::Editable);

  if (out_selection_id) {
    OutputAttribute_Typed<bool> attribute =
        component.attribute_try_get_anonymous_for_output_only<bool>(*out_selection_id,
                                                                    ATTR_DOMAIN_FACE);
    BM_get_selected_faces(bm, attribute.as_span().data());
    // face_map.out
    // boundary_map.out

    attribute.save();
  }

  BM_mesh_free(bm);

  BKE_mesh_normals_tag_dirty(result);

  return result;
}

static void geo_node_extrude_and_move_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");

  geometry_set = bke::geometry_set_realize_instances(geometry_set);

  MeshComponent &mesh_component = geometry_set.get_component_for_write<MeshComponent>();
  if (mesh_component.has_mesh()) {
    const Mesh *input_mesh = mesh_component.get_for_read();

    AttributeDomain domain = ATTR_DOMAIN_POINT;
    int domain_size = input_mesh->totvert;
    if (params.node().custom1 == 1) {
      domain = ATTR_DOMAIN_EDGE;
      domain_size = input_mesh->totedge;
    }
    else if (params.node().custom1 == 2) {
      domain = ATTR_DOMAIN_FACE;
      domain_size = input_mesh->totpoly;
    }
    bke::FieldRef<bool> field = params.get_input_field<bool>("Selection");
    bke::FieldInputs field_inputs = field->prepare_inputs();
    Vector<std::unique_ptr<bke::FieldInputValue>> field_input_values;
    prepare_field_inputs(field_inputs, mesh_component, domain, field_input_values);
    bke::FieldOutput field_output = field->evaluate(IndexRange(domain_size), field_inputs);
    GVArray_Typed<bool> selection_results{field_output.varray_ref()};
    VArray_Span<bool> selection{selection_results};

    AnonymousCustomDataLayerID *out_selection_id = params.output_is_required("Selection") ?
                                                       CustomData_anonymous_id_new("Selection") :
                                                       nullptr;

    const float3 offset = params.extract_input<float3>("Offset");
    Mesh *result;
    if (params.node().custom1 == 1) {
      result = extrude_edges(input_mesh, selection, offset, out_selection_id);
    }
    else if (params.node().custom1 == 2) {
      result = extrude_faces(input_mesh, selection, offset, out_selection_id);
    }
    else {
      result = extrude_vertices(input_mesh, selection, offset, out_selection_id);
    }
    geometry_set.replace_mesh(result);

    if (out_selection_id) {
      params.set_output("Selection",
                        bke::FieldRef<bool>(new bke::AnonymousAttributeField(
                            *out_selection_id, CPPType::get<bool>())));
    }
  }

  params.set_output("Geometry", std::move(geometry_set));
}
}  // namespace blender::nodes

void register_node_type_geo_extrude_and_move()
{
  static bNodeType ntype;

  geo_node_type_base(
      &ntype, GEO_NODE_EXTRUDE_AND_MOVE, "Mesh Extrude And Move", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, geo_node_extrude_and_move_in, geo_node_extrude_and_move_out);
  node_type_init(&ntype, geo_node_extrude_and_move_init);
  ntype.geometry_node_execute = blender::nodes::geo_node_extrude_and_move_exec;
  ntype.draw_buttons = geo_node_extrude_and_move_layout;
  nodeRegisterType(&ntype);
}