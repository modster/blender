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
    {SOCK_STRING, N_("Distance")},
    {SOCK_FLOAT, N_("Distance"), 0.0f, 0, 0, 0, FLT_MIN, FLT_MAX, PROP_DISTANCE},
    {SOCK_STRING, N_("Inset")},
    {SOCK_FLOAT, N_("Inset"), 0.0f, 0, 0, 0, FLT_MIN, FLT_MAX, PROP_DISTANCE},
    {SOCK_BOOLEAN, N_("Individual")},
    {SOCK_STRING, N_("Selection")},
    {SOCK_STRING, N_("Top Face")},
    {SOCK_STRING, N_("Side Face")},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_extrude_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

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

  MeshComponent &mesh_component = geometry_set.get_component_for_write<MeshComponent>();
  if (mesh_component.has_mesh()) {
    const bool default_selection = true;
    GVArray_Typed<bool> selection_attribute = params.get_input_attribute<bool>(
        "Selection", mesh_component, ATTR_DOMAIN_FACE, default_selection);
    VArray_Span<bool> selection{selection_attribute};
    const Mesh *input_mesh = mesh_component.get_for_read();
    // const float distance = params.extract_input<float>("Distance");
    // const float inset = params.extract_input<float>("Inset");

    AttributeDomain attribute_domain = ATTR_DOMAIN_POINT;
    const bool inset_individual_faces = params.extract_input<bool>("Individual");

    if (inset_individual_faces) {
      attribute_domain = ATTR_DOMAIN_FACE;
    }

    const float default_distance = 0;
    GVArray_Typed<float> distance_attribute = params.get_input_attribute<float>(
        "Distance", mesh_component, attribute_domain, default_distance);
    VArray_Span<float> distance{distance_attribute};

    const float default_inset = 0;
    GVArray_Typed<float> inset_attribute = params.get_input_attribute<float>(
        "Inset", mesh_component, attribute_domain, default_inset);
    VArray_Span<float> inset{inset_attribute};

    bool *selection_top_faces_out = nullptr;
    bool *selection_all_faces_out = nullptr;

    Mesh *result = extrude_mesh(input_mesh,
                                selection,
                                distance,
                                inset,
                                inset_individual_faces,
                                &selection_top_faces_out,
                                &selection_all_faces_out);

    const AttributeDomain result_face_domain = ATTR_DOMAIN_FACE;
    std::string selection_top_faces_out_attribute_name = params.get_input<std::string>("Top Face");
    std::string selection_side_faces_out_attribute_name = params.get_input<std::string>(
        "Side Face");
    geometry_set.replace_mesh(result);

    MeshComponent &result_mesh_component = geometry_set.get_component_for_write<MeshComponent>();
    if (!selection_top_faces_out_attribute_name.empty()) {

      OutputAttribute_Typed<bool> selection_top_faces_out_attribute =
          result_mesh_component.attribute_try_get_for_output_only<bool>(
              selection_top_faces_out_attribute_name, result_face_domain);
      Span<bool> selection_faces_top_out_span(selection_top_faces_out, result->totpoly);
      selection_top_faces_out_attribute->set_all(selection_faces_top_out_span);
      selection_top_faces_out_attribute.save();
    }
    if (!selection_side_faces_out_attribute_name.empty()) {
      OutputAttribute_Typed<bool> selection_side_faces_out_attribute =
          result_mesh_component.attribute_try_get_for_output_only<bool>(
              selection_side_faces_out_attribute_name, result_face_domain);
      for (const int i : selection_side_faces_out_attribute->index_range()) {
        if (selection_top_faces_out[i]) {
          selection_all_faces_out[i] = false;
        }
      }
      Span<bool> selection_faces_sides_out_span(selection_all_faces_out, result->totpoly);
      selection_side_faces_out_attribute->set_all(selection_faces_sides_out_span);
      selection_side_faces_out_attribute.save();
    }
    MEM_freeN(selection_top_faces_out);
    MEM_freeN(selection_all_faces_out);
  }

  params.set_output("Geometry", std::move(geometry_set));
}
}  // namespace blender::nodes

void register_node_type_geo_extrude()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_EXTRUDE, "Extrude", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, geo_node_extrude_in, geo_node_extrude_out);
  node_type_init(&ntype, geo_node_extrude_init);
  node_type_update(&ntype, geo_node_extrude_update);
  ntype.draw_buttons = geo_node_extrude_layout;
  ntype.geometry_node_execute = blender::nodes::geo_node_extrude_exec;
  nodeRegisterType(&ntype);
}
