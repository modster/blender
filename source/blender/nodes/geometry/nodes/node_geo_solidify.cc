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

#include "BKE_node.h"
#include "BKE_solidifiy.h"

#include "DNA_mesh_types.h"
#include "DNA_modifier_types.h"
#include "DNA_node_types.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "node_geometry_util.hh"

static bNodeSocketTemplate geo_node_solidify_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_FLOAT, N_("Thickness"), 0.1f, 0.0f, 0.0f, 0.0f, -FLT_MAX, FLT_MAX},
    {SOCK_FLOAT, N_("Clamp"), 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.0f},
    {SOCK_FLOAT, N_("Offset"), -1.0f, 0.0f, 0.0f, 0.0f, -1.0f, 1.0f},
    {SOCK_BOOLEAN, N_("Fill"), true},
    {SOCK_BOOLEAN, N_("Rim"), true},
    {SOCK_STRING, N_("Distance")},
    {SOCK_STRING, N_("Shell Verts")},
    {SOCK_STRING, N_("Rim Verts")},
    {SOCK_STRING, N_("Shell Faces")},
    {SOCK_STRING, N_("Rim Faces")},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_solidify_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

namespace blender::nodes {

static void geo_node_solidify_init(bNodeTree *UNUSED(tree), bNode *node)
{
  NodeGeometrySolidify *node_storage = (NodeGeometrySolidify *)MEM_callocN(
      sizeof(NodeGeometrySolidify), __func__);

  node_storage->mode = MOD_SOLIDIFY_MODE_NONMANIFOLD;
  node->storage = node_storage;
}

static void geo_node_solidify_update(bNodeTree *UNUSED(ntree), bNode *UNUSED(node))
{
}

static void geo_node_solidify_exec(GeoNodeExecParams params)
{
  const bNode &node = params.node();
  NodeGeometrySolidify &node_storage = *(NodeGeometrySolidify *)node.storage;
  const Object *self_object = params.self_object();

  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");
  bool add_fill = params.extract_input<bool>("Fill");
  bool add_rim = params.extract_input<bool>("Rim");
  const std::string Distance_name = params.extract_input<std::string>("Distance");

  char flag = 0;

  if (add_fill) {
    flag |= MOD_SOLIDIFY_SHELL;
  }

  if (add_rim) {
    flag |= MOD_SOLIDIFY_RIM;
  }

  float thickness = params.extract_input<float>("Thickness");
  float offset = params.extract_input<float>("Offset");
  float offset_clamp = params.extract_input<float>("Clamp");

  geometry_set = geometry_set_realize_instances(geometry_set);

  if (geometry_set.has<MeshComponent>()) {
    MeshComponent &mesh_component = geometry_set.get_component_for_write<MeshComponent>();
    Mesh *input_mesh = mesh_component.get_for_write();
    Mesh *output_mesh;

    GVArray_Typed<float> vertex_mask = mesh_component.attribute_get_for_read<float>(
        Distance_name, ATTR_DOMAIN_POINT, 1.0f);

    float *distance = (float *)MEM_callocN(sizeof(float) * (unsigned long)input_mesh->totvert,
                                           __func__);

    for (int i : vertex_mask.index_range()) {
      distance[i] = vertex_mask[i];
    }

    SolidifyData solidify_node_data = {
        self_object,
        "",
        "",
        "",
        thickness,
        offset,
        0.0f,
        offset_clamp,
        MOD_SOLIDIFY_MODE_NONMANIFOLD,
        node_storage.nonmanifold_offset_mode,
        node_storage.nonmanifold_boundary_mode,
        0.0f,
        0.0f,
        0.0f,
        flag,
        0,
        0,
        0.01f,
        0.0f,
        distance,
    };

    bool *shell_verts = nullptr;
    bool *rim_verts = nullptr;
    bool *shell_faces = nullptr;
    bool *rim_faces = nullptr;

    output_mesh = solidify_nonmanifold(&solidify_node_data, input_mesh, &shell_verts, &rim_verts, &shell_faces, &rim_faces);

    geometry_set.replace_mesh(output_mesh);

    const AttributeDomain result_point_domain = ATTR_DOMAIN_POINT;

    const std::string shell_verts_attribute_name = params.get_input<std::string>("Shell Verts");
    OutputAttribute_Typed<bool> shell_verts_attribute =
        mesh_component.attribute_try_get_for_output_only<bool>(shell_verts_attribute_name,
                                                               result_point_domain);

    const std::string rim_verts_attribute_name = params.get_input<std::string>("Rim Verts");
    OutputAttribute_Typed<bool> rim_verts_attribute =
        mesh_component.attribute_try_get_for_output_only<bool>(rim_verts_attribute_name,
                                                               result_point_domain);

    const AttributeDomain result_face_domain = ATTR_DOMAIN_FACE;

    const std::string shell_faces_attribute_name = params.get_input<std::string>("Shell Faces");
    OutputAttribute_Typed<bool> shell_faces_attribute =
        mesh_component.attribute_try_get_for_output_only<bool>(shell_faces_attribute_name,
                                                               result_face_domain);

    const std::string rim_faces_attribute_name = params.get_input<std::string>("Rim Faces");
    OutputAttribute_Typed<bool> rim_faces_attribute =
        mesh_component.attribute_try_get_for_output_only<bool>(rim_faces_attribute_name,
                                                               result_face_domain);

    if (solidify_node_data.flag & MOD_SOLIDIFY_SHELL) {
      if(!shell_verts_attribute_name.empty()){
        MutableSpan<bool> shell_verts_span = shell_verts_attribute.as_span();
        for (const int i : shell_verts_span.index_range()) {
          shell_verts_span[i] = shell_verts[i];
        }
        shell_verts_attribute.save();
      }
      if(!shell_faces_attribute_name.empty()){
        MutableSpan<bool> shell_faces_span = shell_faces_attribute.as_span();
        for (const int i : shell_faces_span.index_range()) {
          shell_faces_span[i] = shell_faces[i];
        }
        shell_faces_attribute.save();
      }
    }

    if (solidify_node_data.flag & MOD_SOLIDIFY_RIM) {
      if(!rim_verts_attribute_name.empty()) {
        MutableSpan<bool> rim_verts_span = rim_verts_attribute.as_span();
        for (const int i : rim_verts_span.index_range()) {
          rim_verts_span[i] = rim_verts[i];
        }
        rim_verts_attribute.save();
      }
      if(!rim_faces_attribute_name.empty()) {
        MutableSpan<bool> rim_faces_span = rim_faces_attribute.as_span();
        for (const int i : rim_faces_span.index_range()) {
          rim_faces_span[i] = rim_faces[i];
        }
        rim_faces_attribute.save();
      }
    }
    MEM_freeN(distance);
    MEM_freeN(shell_verts);
    MEM_freeN(rim_verts);
    MEM_freeN(shell_faces);
    MEM_freeN(rim_faces);
  }
  params.set_output("Geometry", geometry_set);
}

static void geo_node_solidify_layout(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
{
  uiLayoutSetPropSep(layout, true);
  uiLayoutSetPropDecorate(layout, false);
  uiItemR(layout, ptr, "nonmanifold_offset_mode", 0, nullptr, ICON_NONE);
  uiItemR(layout, ptr, "nonmanifold_boundary_mode", 0, nullptr, ICON_NONE);
}

}  // namespace blender::nodes

void register_node_type_geo_solidify()
{
  static bNodeType ntype;
  geo_node_type_base(&ntype, GEO_NODE_SOLIDIFY, "Solidify", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, geo_node_solidify_in, geo_node_solidify_out);
  node_type_storage(
      &ntype, "NodeGeometrySolidify", node_free_standard_storage, node_copy_standard_storage);
  node_type_init(&ntype, blender::nodes::geo_node_solidify_init);
  node_type_size(&ntype, 167, 100, 600);
  node_type_update(&ntype, blender::nodes::geo_node_solidify_update);
  ntype.geometry_node_execute = blender::nodes::geo_node_solidify_exec;
  ntype.draw_buttons = blender::nodes::geo_node_solidify_layout;
  nodeRegisterType(&ntype);
}
