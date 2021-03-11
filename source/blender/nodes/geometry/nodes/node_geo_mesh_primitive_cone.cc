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

#include "BLI_map.hh"
#include "BLI_math_matrix.h"

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"

#include "BKE_mesh.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "node_geometry_util.hh"

static bNodeSocketTemplate geo_node_mesh_primitive_cone_in[] = {
    {SOCK_INT, N_("Vertices"), 32, 0.0f, 0.0f, 0.0f, 3, 4096},
    {SOCK_FLOAT, N_("Radius Top"), 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, FLT_MAX, PROP_DISTANCE},
    {SOCK_FLOAT, N_("Radius Bottom"), 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, FLT_MAX, PROP_DISTANCE},
    {SOCK_FLOAT, N_("Depth"), 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, FLT_MAX, PROP_DISTANCE},
    {SOCK_VECTOR, N_("Location"), 0.0f, 0.0f, 0.0f, 1.0f, -FLT_MAX, FLT_MAX, PROP_TRANSLATION},
    {SOCK_VECTOR, N_("Rotation"), 0.0f, 0.0f, 0.0f, 1.0f, -FLT_MAX, FLT_MAX, PROP_EULER},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_mesh_primitive_cone_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

static void geo_node_mesh_primitive_cone_layout(uiLayout *layout,
                                                bContext *UNUSED(C),
                                                PointerRNA *ptr)
{
  uiLayoutSetPropSep(layout, true);
  uiLayoutSetPropDecorate(layout, false);
  uiItemR(layout, ptr, "fill_type", 0, IFACE_("Fill"), ICON_NONE);
}

static void geo_node_mesh_primitive_cone_init(bNodeTree *UNUSED(ntree), bNode *node)
{
  NodeGeometryMeshCone *node_storage = (NodeGeometryMeshCone *)MEM_callocN(
      sizeof(NodeGeometryMeshCone), __func__);

  node_storage->fill_type = GEO_NODE_MESH_CIRCLE_FILL_NGON;

  node->storage = node_storage;
}

namespace blender::nodes {

static void geo_node_mesh_primitive_cone_exec(GeoNodeExecParams params)
{
  const bNode &node = params.node();
  const NodeGeometryMeshCone &storage = *(const NodeGeometryMeshCone *)node.storage;

  const GeometryNodeMeshCircleFillType fill_type = (const GeometryNodeMeshCircleFillType)
                                                       storage.fill_type;

  const int verts_num = params.extract_input<int>("Vertices");
  if (verts_num < 3) {
    params.set_output("Geometry", GeometrySet());
    return;
  }

  const float radius_top = params.extract_input<float>("Radius Top");
  const float radius_bottom = params.extract_input<float>("Radius Bottom");
  const float depth = params.extract_input<float>("Depth");
  const float3 location = params.extract_input<float3>("Location");
  const float3 rotation = params.extract_input<float3>("Rotation");

  Mesh *mesh = create_cylinder_or_cone_mesh(
      radius_top, radius_bottom, depth, verts_num, fill_type);

  BLI_assert(BKE_mesh_is_valid(mesh));

  /* Transform the mesh so that the base of the cone is at the origin. */
  transform_mesh(mesh, location + float3(0.0f, 0.0f, depth), rotation, float3(1));

  params.set_output("Geometry", GeometrySet::create_with_mesh(mesh));
}

}  // namespace blender::nodes

void register_node_type_geo_mesh_primitive_cone()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_MESH_PRIMITIVE_CONE, "Cone", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(
      &ntype, geo_node_mesh_primitive_cone_in, geo_node_mesh_primitive_cone_out);
  node_type_init(&ntype, geo_node_mesh_primitive_cone_init);
  node_type_storage(
      &ntype, "NodeGeometryMeshCone", node_free_standard_storage, node_copy_standard_storage);
  ntype.geometry_node_execute = blender::nodes::geo_node_mesh_primitive_cone_exec;
  ntype.draw_buttons = geo_node_mesh_primitive_cone_layout;
  nodeRegisterType(&ntype);
}
