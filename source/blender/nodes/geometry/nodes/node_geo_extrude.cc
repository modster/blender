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
    {SOCK_FLOAT, N_("Distance"), 0.0f, 0, 0, 0, FLT_MIN, FLT_MAX, PROP_DISTANCE},
    {SOCK_FLOAT, N_("Inset"), 0.0f, 0, 0, 0, FLT_MIN, FLT_MAX, PROP_DISTANCE},
    {SOCK_BOOLEAN, N_("Individual")},
    {SOCK_BOOLEAN, N_("Selection"), 1, 0, 0, 0, 0, 0, PROP_NONE, SOCK_HIDE_VALUE},
    //{SOCK_STRING, N_("Top Face")},
    //{SOCK_STRING, N_("Side Face")},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_extrude_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

<<<<<<< HEAD
=======
static void geo_node_extrude_layout(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
{
  uiLayoutSetPropSep(layout, true);
  uiLayoutSetPropDecorate(layout, false);
  uiItemR(layout, ptr, "distance_mode", 0, nullptr, ICON_NONE);
  uiItemR(layout, ptr, "inset_mode", 0, nullptr, ICON_NONE);
}

static void geo_node_extrude_init(bNodeTree *UNUSED(tree), bNode *node)
{
  node->custom1 = GEO_NODE_ATTRIBUTE_INPUT_FLOAT;
  node->custom2 = GEO_NODE_ATTRIBUTE_INPUT_FLOAT;
}

static void geo_node_extrude_update(bNodeTree *UNUSED(ntree), bNode *node)
{
  blender::nodes::update_attribute_input_socket_availabilities(
      *node, "Distance", (GeometryNodeAttributeInputMode)node->custom1, true);
  blender::nodes::update_attribute_input_socket_availabilities(
      *node, "Inset", (GeometryNodeAttributeInputMode)node->custom2, true);
}

>>>>>>> soc-2021-porting-modifiers-to-nodes-extrude
using blender::Span;

static Mesh *extrude_mesh(const Mesh *mesh,
                          const Span<bool> selection,
                          const float distance,
                          const float inset,
                          const bool inset_individual_faces)
{
  BLI_assert(selection.size() == mesh->totpoly);
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
                 "inset_individual faces=%hf use_even_offset=%b thickness=%f depth=%f",
                 BM_ELEM_SELECT,
                 true,
                 inset,
                 distance);
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
                 "inset_region faces=%hf use_boundary=%b use_even_offset=%b thickness=%f depth=%f",
                 BM_ELEM_SELECT,
                 true,
                 true,
                 inset,
                 distance);
  }
  BMO_op_exec(bm, &op);
  BMO_op_finish(bm, &op);

  CustomData_MeshMasks cd_mask_extra = {};
  Mesh *result = BKE_mesh_from_bmesh_for_eval_nomain(bm, &cd_mask_extra, mesh);
  BM_mesh_free(bm);

  result->runtime.cd_dirty_vert |= CD_MASK_NORMAL;
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
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");

  geometry_set = geometry_set_realize_instances(geometry_set);

  const MeshComponent &mesh_component = geometry_set.get_component_for_write<MeshComponent>();
  if (mesh_component.has_mesh()) {
    bke::FieldRef<bool> field = params.get_input_field<bool>("Selection");
    bke::FieldInputs field_inputs = field->prepare_inputs();
    Vector<std::unique_ptr<bke::FieldInputValue>> field_input_values;
    prepare_field_inputs(field_inputs, mesh_component, ATTR_DOMAIN_FACE, field_input_values);
    bke::FieldOutput field_output = field->evaluate(
        IndexRange(mesh_component.attribute_domain_size(ATTR_DOMAIN_FACE)), field_inputs);

    GVArray_Typed<bool> selection{field_output.varray_ref()};
    VArray_Span<bool> selection_span{selection};

    const Mesh *input_mesh = mesh_component.get_for_read();
    const float distance = params.extract_input<float>("Distance");
    const float inset = params.extract_input<float>("Inset");
    const bool inset_individual_faces = params.extract_input<bool>("Individual");
    Mesh *result = extrude_mesh(
        input_mesh, selection_span, distance, inset, inset_individual_faces);
    geometry_set.replace_mesh(result);
  }

  params.set_output("Geometry", std::move(geometry_set));
}
}  // namespace blender::nodes

void register_node_type_geo_extrude()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_EXTRUDE, "Mesh Extrude", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, geo_node_extrude_in, geo_node_extrude_out);
  ntype.geometry_node_execute = blender::nodes::geo_node_extrude_exec;
  nodeRegisterType(&ntype);
}
