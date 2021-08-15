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

#include "node_geometry_util.hh"

static bNodeSocketTemplate geo_node_mesh_extrude_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_VECTOR, N_("Offset"), 0.0f, 0.0f, 1.0f, 0.0f, -FLT_MAX, FLT_MAX, PROP_TRANSLATION},
    {SOCK_STRING, N_("Selection")},
    {SOCK_STRING, N_("Out Selection")},

    {-1, ""},
};

static bNodeSocketTemplate geo_node_mesh_extrude_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

static void geo_node_mesh_extrude_layout(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
{
  uiItemR(layout, ptr, "extrude_mode", 0, "", ICON_NONE);
}

static void geo_node_mesh_extrude_init(bNodeTree *UNUSED(ntree), bNode *node)
{
  node->custom1 = 0;
}

namespace blender::nodes {

static void SetOutputSelection(const std::string &selection_out_attribute_name,
                               BMesh *bm,
                               Mesh *result)
{
  MeshComponent component;
  component.replace(result, GeometryOwnershipType::Editable);

  if (!selection_out_attribute_name.empty()) {
    bke::OutputAttribute_Typed<bool> attribute = component.attribute_try_get_for_output_only<bool>(
        selection_out_attribute_name, ATTR_DOMAIN_POINT);
    BM_get_selected_vertices(bm, attribute.as_span().data());
    attribute.save();
  }
}

static Mesh *extrude_vertices(const Mesh *mesh,
                              const Span<bool> selection,
                              const float3 offset,
                              const std::string selection_out_attribute_name)
{
  const BMeshCreateParams bmesh_create_params = {true};
  const BMeshFromMeshParams bmesh_from_mesh_params = {
      true, 0, 0, 0, {CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX}};

  BMesh *bm = BKE_mesh_to_bmesh_ex(mesh, &bmesh_create_params, &bmesh_from_mesh_params);

  BM_select_vertices(bm, selection.data());

  BMOperator extrude_op;
  BMO_op_initf(bm, &extrude_op, 0, "extrude_vert_indiv verts=%hv", BM_ELEM_SELECT);
  BMO_op_exec(bm, &extrude_op);

  float _offset[3] = {offset.x, offset.y, offset.z};
  BMOperator move_op;

  BM_mesh_elem_hflag_disable_all(bm, BM_VERT, BM_ELEM_SELECT, false);
  BMO_slot_buffer_hflag_enable(
      bm, extrude_op.slots_out, "verts.out", BM_VERT, BM_ELEM_SELECT, false);

  BMO_op_initf(bm, &move_op, 0, "translate vec=%v verts=%hv", _offset, BM_ELEM_SELECT);

  BMO_op_exec(bm, &move_op);
  BMO_op_finish(bm, &move_op);
  BMO_op_finish(bm, &extrude_op);

  CustomData_MeshMasks cd_mask_extra = {CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX};
  Mesh *result = BKE_mesh_from_bmesh_for_eval_nomain(bm, &cd_mask_extra, mesh);

  SetOutputSelection(selection_out_attribute_name, bm, result);

  BM_mesh_free(bm);
  BKE_mesh_normals_tag_dirty(result);

  return result;
}

static Mesh *extrude_edges(const Mesh *mesh,
                           const Span<bool> selection,
                           const float3 offset,
                           const std::string selection_out_attribute_name)
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

  float _offset[3] = {offset.x, offset.y, offset.z};
  BMOperator move_op;

  BM_mesh_elem_hflag_disable_all(bm, BM_VERT, BM_ELEM_SELECT, false);
  BMO_slot_buffer_hflag_enable(
      bm, extrude_op.slots_out, "geom.out", BM_VERT, BM_ELEM_SELECT, false);

  BMO_op_initf(bm, &move_op, 0, "translate vec=%v verts=%hv", _offset, BM_ELEM_SELECT);

  BMO_op_exec(bm, &move_op);
  BMO_op_finish(bm, &move_op);
  BMO_op_finish(bm, &extrude_op);

  CustomData_MeshMasks cd_mask_extra = {CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX};
  Mesh *result = BKE_mesh_from_bmesh_for_eval_nomain(bm, &cd_mask_extra, mesh);

  SetOutputSelection(selection_out_attribute_name, bm, result);

  BM_mesh_free(bm);
  BKE_mesh_normals_tag_dirty(result);

  return result;
}

static Mesh *extrude_faces(const Mesh *mesh,
                           const Span<bool> selection,
                           const float3 offset,
                           std::string selection_out_attribute_name)
{
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

  SetOutputSelection(selection_out_attribute_name, bm, result);

  BM_mesh_free(bm);
  BKE_mesh_normals_tag_dirty(result);

  return result;
}

static void geo_node_mesh_extrude_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");

  geometry_set = bke::geometry_set_realize_instances(geometry_set);
  MeshComponent &mesh_component = geometry_set.get_component_for_write<MeshComponent>();

  const float3 offset = params.extract_input<float3>("Offset");

  if (offset.length() > 0 && mesh_component.has_mesh()) {
    const Mesh *input_mesh = mesh_component.get_for_read();

    AttributeDomain domain = ATTR_DOMAIN_POINT;
    if (params.node().custom1 == 1) {
      domain = ATTR_DOMAIN_EDGE;
    }
    else if (params.node().custom1 == 2) {
      domain = ATTR_DOMAIN_FACE;
    }

    const bool default_selection = true;
    GVArray_Typed<bool> selection_attribute = params.get_input_attribute<bool>(
        "Selection", mesh_component, domain, default_selection);
    VArray_Span<bool> selection{selection_attribute};

    std::string out_selection_attribute_name = params.get_input<std::string>("Out Selection");

    Mesh *result;
    if (params.node().custom1 == 1) {
      result = extrude_edges(input_mesh, selection, offset, out_selection_attribute_name);
    }
    else if (params.node().custom1 == 2) {
      result = extrude_faces(input_mesh, selection, offset, out_selection_attribute_name);
    }
    else {
      result = extrude_vertices(input_mesh, selection, offset, out_selection_attribute_name);
    }
    geometry_set.replace_mesh(result);
  }

  params.set_output("Geometry", std::move(geometry_set));
}
}  // namespace blender::nodes

void register_node_type_geo_mesh_extrude()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_MESH_EXTRUDE, "Mesh Extrude", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, geo_node_mesh_extrude_in, geo_node_mesh_extrude_out);
  node_type_init(&ntype, geo_node_mesh_extrude_init);
  ntype.geometry_node_execute = blender::nodes::geo_node_mesh_extrude_exec;
  ntype.draw_buttons = geo_node_mesh_extrude_layout;
  nodeRegisterType(&ntype);
}
