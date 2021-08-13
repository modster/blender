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

static bNodeSocketTemplate geo_node_mesh_inset_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_STRING, N_("Distance")},
    {SOCK_FLOAT, N_("Distance"), 0.0f, 0, 0, 0, FLT_MIN, FLT_MAX, PROP_DISTANCE},
    {SOCK_STRING, N_("mesh_inset")},
    {SOCK_FLOAT, N_("mesh_inset"), 0.0f, 0, 0, 0, FLT_MIN, FLT_MAX, PROP_DISTANCE},
    {SOCK_BOOLEAN, N_("Individual")},
    {SOCK_STRING, N_("Selection")},
    {SOCK_STRING, N_("Top Face")},
    {SOCK_STRING, N_("Side Face")},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_mesh_inset_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

static void geo_node_mesh_inset_layout(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
{
  uiLayoutSetPropSep(layout, true);
  uiLayoutSetPropDecorate(layout, false);
  uiItemR(layout, ptr, "distance_mode", 0, nullptr, ICON_NONE);
  uiItemR(layout, ptr, "inset_mode", 0, nullptr, ICON_NONE);
}

static void geo_node_mesh_inset_init(bNodeTree *UNUSED(tree), bNode *node)
{
  node->custom1 = GEO_NODE_ATTRIBUTE_INPUT_FLOAT;
  node->custom2 = GEO_NODE_ATTRIBUTE_INPUT_FLOAT;
}

static void geo_node_mesh_inset_update(bNodeTree *UNUSED(ntree), bNode *node)
{
  blender::nodes::update_attribute_input_socket_availabilities(
      *node, "Distance", (GeometryNodeAttributeInputMode)node->custom1, true);
  blender::nodes::update_attribute_input_socket_availabilities(
      *node, "mesh_inset", (GeometryNodeAttributeInputMode)node->custom2, true);
}

using blender::Span;

static Mesh *mesh_inset_mesh(const Mesh *mesh,
                             const Span<bool> selection,
                             const Span<float> distance,
                             const Span<float> mesh_inset,
                             const bool mesh_inset_individual_faces,
                             std::string selection_top_faces_out_attribute_name,
                             std::string selection_side_faces_out_attribute_name)
{
  const BMeshCreateParams bmesh_create_params = {true};
  const BMeshFromMeshParams bmesh_from_mesh_params = {
      true, 0, 0, 0, {CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX}};

  BMesh *bm = BKE_mesh_to_bmesh_ex(mesh, &bmesh_create_params, &bmesh_from_mesh_params);

  BM_select_faces(bm, selection.data());
  BMOperator op;
  if (mesh_inset_individual_faces) {
    BMO_op_initf(bm,
                 &op,
                 0,
                 "inset_individual faces=%hf use_even_offset=%b thickness=%f depth=%f "
                 "thickness_array=%p depth_array=%p use_attributes=%b",
                 BM_ELEM_SELECT,
                 true,
                 mesh_inset[0],
                 distance[0],
                 mesh_inset.data(),
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
                 mesh_inset[0],
                 distance[0],
                 mesh_inset.data(),
                 distance.data(),
                 true);
  }
  BM_tag_new_faces(bm, &op);
  BM_untag_faces_by_tag(bm, BM_ELEM_SELECT);

  BMO_op_exec(bm, &op);
  BMO_op_finish(bm, &op);

  CustomData_MeshMasks cd_mask_extra = {CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX};

  Mesh *result = BKE_mesh_from_bmesh_for_eval_nomain(bm, &cd_mask_extra, mesh);

  MeshComponent component;
  component.replace(result, GeometryOwnershipType::Editable);

  if (!selection_top_faces_out_attribute_name.empty()) {
    blender::bke::OutputAttribute_Typed<bool> attribute =
        component.attribute_try_get_for_output_only<bool>(selection_top_faces_out_attribute_name,
                                                          ATTR_DOMAIN_FACE);
    BM_get_selected_faces(bm, attribute.as_span().data());
    attribute.save();
  }

  if (!selection_side_faces_out_attribute_name.empty()) {
    blender::bke::OutputAttribute_Typed<bool> attribute =
        component.attribute_try_get_for_output_only<bool>(selection_side_faces_out_attribute_name,
                                                          ATTR_DOMAIN_FACE);
    BM_get_tagged_faces(bm, attribute.as_span().data());
    attribute.save();
  }

  BM_mesh_free(bm);
  BKE_mesh_normals_tag_dirty(result);

  return result;
}

namespace blender::nodes {
static void geo_node_mesh_inset_exec(GeoNodeExecParams params)
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

    AttributeDomain attribute_domain = ATTR_DOMAIN_POINT;
    const bool mesh_inset_individual_faces = params.extract_input<bool>("Individual");

    if (mesh_inset_individual_faces) {
      attribute_domain = ATTR_DOMAIN_FACE;
    }

    const float default_distance = 0;
    GVArray_Typed<float> distance_attribute = params.get_input_attribute<float>(
        "Distance", mesh_component, attribute_domain, default_distance);
    VArray_Span<float> distance{distance_attribute};

    const float default_mesh_inset = 0;
    GVArray_Typed<float> mesh_inset_attribute = params.get_input_attribute<float>(
        "mesh_inset", mesh_component, attribute_domain, default_mesh_inset);
    VArray_Span<float> mesh_inset{mesh_inset_attribute};

    std::string selection_top_faces_out_attribute_name = params.get_input<std::string>("Top Face");
    std::string selection_side_faces_out_attribute_name = params.get_input<std::string>(
        "Side Face");

    Mesh *result = mesh_inset_mesh(input_mesh,
                                   selection,
                                   distance,
                                   mesh_inset,
                                   mesh_inset_individual_faces,
                                   selection_top_faces_out_attribute_name,
                                   selection_side_faces_out_attribute_name);

    geometry_set.replace_mesh(result);
  }

  params.set_output("Geometry", std::move(geometry_set));
}
}  // namespace blender::nodes

void register_node_type_geo_mesh_inset()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_MESH_INSET, "MeshInset", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, geo_node_mesh_inset_in, geo_node_mesh_inset_out);
  node_type_init(&ntype, geo_node_mesh_inset_init);
  node_type_update(&ntype, geo_node_mesh_inset_update);
  ntype.draw_buttons = geo_node_mesh_inset_layout;
  ntype.geometry_node_execute = blender::nodes::geo_node_mesh_inset_exec;
  nodeRegisterType(&ntype);
}
