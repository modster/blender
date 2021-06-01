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

#include "BKE_solidifiy.h"
#include "BKE_node.h"

#include "DNA_modifier_types.h"
#include "DNA_node_types.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "node_geometry_util.hh"

static bNodeSocketTemplate geo_node_solidify_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_FLOAT, N_("Thickness"), 0.1f, 0.0f, 0.0f, 0.0f, -FLT_MAX, FLT_MAX},
    {SOCK_FLOAT, N_("Clamp Thickness"), 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.0f},
    {SOCK_FLOAT, N_("Offset"), -1.0f, 0.0f, 0.0f, 0.0f, -1.0f, 1.0f},
    {SOCK_BOOLEAN, N_("Fill"), true},
    {SOCK_BOOLEAN, N_("Rim Only"), false},
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

  node_storage->mode = MOD_SOLIDIFY_MODE_EXTRUDE;
  node->storage = node_storage;
}

static void geo_node_solidify_update(bNodeTree *UNUSED(ntree), bNode *node)
{
  NodeGeometrySolidify &node_storage = *(NodeGeometrySolidify *)node->storage;

  //update_attribute_input_socket_availabilities(
  //    *node, "Translation", (GeometryNodeAttributeInputMode)node_storage.input_type);
}

static void geo_node_solidify_exec(GeoNodeExecParams params)
{
  const bNode &node = params.node();
  NodeGeometrySolidify &node_storage = *(NodeGeometrySolidify *)node.storage;

  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");
  bool add_fill = params.extract_input<bool>("Fill");
  bool add_rim_only = params.extract_input<bool>("Rim Only");

  char flag = 0;

  if(add_fill) {
    flag |= MOD_SOLIDIFY_RIM;
  }

  if(add_rim_only){
    flag |= MOD_SOLIDIFY_NOSHELL;
  }

  float thickness = params.extract_input<float>("Thickness");
  float offset = params.extract_input<float>("Offset");
  float offset_clamp = params.extract_input<float>("Clamp Thickness");

  geometry_set = geometry_set_realize_instances(geometry_set);

  if (geometry_set.has<MeshComponent>()) {
    SolidifyData solidify_node_data = {
      "",
      "",
      "",
      thickness,
      offset,
      0.0f,
      offset_clamp,
      node_storage.mode,
        MOD_SOLIDIFY_NONMANIFOLD_OFFSET_MODE_FIXED,
      MOD_SOLIDIFY_NONMANIFOLD_BOUNDARY_MODE_NONE,
        0.0f,
      0.0f,
      0.0f,
      flag,
      0,
      0,
      0.01f,
      0.0f,
    };

    MeshComponent &meshComponent = geometry_set.get_component_for_write<MeshComponent>();
    Mesh *input_mesh = meshComponent.get_for_write();
    Mesh *return_mesh;

    if(node_storage.mode == MOD_SOLIDIFY_MODE_EXTRUDE){
      return_mesh = solidify_extrude(&solidify_node_data, input_mesh);
    }else{
      return_mesh = solidify_nonmanifold(&solidify_node_data, input_mesh);
    }
    geometry_set.replace_mesh(return_mesh);
  }

  params.set_output("Geometry", geometry_set);
}

static void geo_node_solidify_layout(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
{
  uiItemR(layout, ptr, "mode", UI_ITEM_R_EXPAND, nullptr, ICON_NONE);
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
  node_type_update(&ntype, blender::nodes::geo_node_solidify_update);
  ntype.geometry_node_execute = blender::nodes::geo_node_solidify_exec;
  ntype.draw_buttons = blender::nodes::geo_node_solidify_layout;
  nodeRegisterType(&ntype);
}
