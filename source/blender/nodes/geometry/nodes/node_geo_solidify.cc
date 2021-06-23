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

/*extern "C" {    // another way
  Mesh *solidify_extrude_modifyMesh( Mesh *mesh);
};*/

static bNodeSocketTemplate geo_node_solidify_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_STRING, N_("Thickness")},
    {SOCK_FLOAT, N_("Thickness"), 0.1f, 0.0f, 0.0f, 0.0f, -FLT_MAX, FLT_MAX, PROP_DISTANCE},
    {SOCK_FLOAT, N_("Clamp"), 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.0f},
    {SOCK_FLOAT, N_("Offset"), -1.0f, 0.0f, 0.0f, 0.0f, -1.0f, 1.0f, PROP_FACTOR},
    {SOCK_BOOLEAN, N_("Fill"), true},
    {SOCK_BOOLEAN, N_("Rim"), true},
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

  node->storage = node_storage;
}

static void geo_node_solidify_layout(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
{
  uiLayoutSetPropSep(layout, true);
  uiLayoutSetPropDecorate(layout, false);
  uiItemR(layout, ptr, "thickness_mode", 0, nullptr, ICON_NONE);
  uiItemR(layout, ptr, "nonmanifold_offset_mode", 0, nullptr, ICON_NONE);
  uiItemR(layout, ptr, "nonmanifold_boundary_mode", 0, nullptr, ICON_NONE);
}

static void geo_node_solidify_update(bNodeTree *UNUSED(ntree), bNode *node)
{
  const NodeGeometrySolidify *node_storage = (NodeGeometrySolidify *)node->storage;

  update_attribute_input_socket_availabilities(
      *node, "Thickness", (GeometryNodeAttributeInputMode)node_storage->thickness_mode, true);
}

static void geo_node_solidify_exec(GeoNodeExecParams params)
{
  const bNode &node = params.node();
  NodeGeometrySolidify &node_storage = *(NodeGeometrySolidify *)node.storage;
  const Object *self_object = params.self_object();

  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");
  bool add_fill = params.extract_input<bool>("Fill");
  bool add_rim = params.extract_input<bool>("Rim");

  char flag = 0;

  if (add_fill) {
    flag |= MOD_SOLIDIFY_SHELL;
  }

  if (add_rim) {
    flag |= MOD_SOLIDIFY_RIM;
  }

  float offset = params.extract_input<float>("Offset");
  float offset_clamp = params.extract_input<float>("Clamp");

  geometry_set = geometry_set_realize_instances(geometry_set);

  if (geometry_set.has<MeshComponent>()) {
    MeshComponent &mesh_component = geometry_set.get_component_for_write<MeshComponent>();
    Mesh *input_mesh = mesh_component.get_for_write();
    Mesh *output_mesh;

    GVArrayPtr thickness = params.get_input_attribute(
        "Thickness", mesh_component, ATTR_DOMAIN_POINT, CD_PROP_FLOAT, nullptr);
    if (!thickness) {
      return;
    }
    float *distance = (float *)MEM_callocN(sizeof(float) * (unsigned long)input_mesh->totvert,
                                           __func__);

    for (int i : thickness->typed<float>().index_range()) {
      distance[i] = thickness->typed<float>()[i];
    }

    SolidifyData solidify_node_data = {
        self_object,
        1,
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
    const AttributeDomain result_face_domain = ATTR_DOMAIN_FACE;

    const std::string shell_faces_attribute_name = params.get_input<std::string>("Shell Faces");
    const std::string rim_faces_attribute_name = params.get_input<std::string>("Rim Faces");


    if (solidify_node_data.flag & MOD_SOLIDIFY_SHELL) {
      if(!shell_faces_attribute_name.empty()){
        OutputAttribute_Typed<bool> shell_faces_attribute =
            mesh_component.attribute_try_get_for_output_only<bool>(shell_faces_attribute_name,
                                                                   result_face_domain);
        Span<bool> s(shell_faces, shell_faces_attribute->size());
        shell_faces_attribute->set_all(s);
      }
    }

    if (solidify_node_data.flag & MOD_SOLIDIFY_RIM) {
      if(!rim_faces_attribute_name.empty()) {
        OutputAttribute_Typed<bool> rim_faces_attribute =
            mesh_component.attribute_try_get_for_output_only<bool>(rim_faces_attribute_name,
                                                                   result_face_domain);
        Span<bool> s(shell_faces, rim_faces_attribute->size());
        rim_faces_attribute->set_all(s);
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
