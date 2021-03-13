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

static bNodeSocketTemplate geo_node_mesh_primitive_line_in[] = {
    // {SOCK_INT, N_("Vertices"), 32, 0.0f, 0.0f, 0.0f, 3, 4096},
    // {SOCK_FLOAT, N_("Radius"), 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, FLT_MAX, PROP_DISTANCE},
    // {SOCK_VECTOR, N_("Location"), 0.0f, 0.0f, 0.0f, 1.0f, -FLT_MAX, FLT_MAX, PROP_TRANSLATION},
    // {SOCK_VECTOR, N_("Rotation"), 0.0f, 0.0f, 0.0f, 1.0f, -FLT_MAX, FLT_MAX, PROP_EULER},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_mesh_primitive_line_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

static void geo_node_mesh_primitive_line_layout(uiLayout *layout,
                                                bContext *UNUSED(C),
                                                PointerRNA *ptr)
{
  uiLayoutSetPropSep(layout, true);
  uiLayoutSetPropDecorate(layout, false);
  uiItemR(layout, ptr, "mode", 0, nullptr, ICON_NONE);
}

static void geo_node_mesh_primitive_line_init(bNodeTree *UNUSED(ntree), bNode *node)
{
  NodeGeometryMeshLine *node_storage = (NodeGeometryMeshLine *)MEM_callocN(
      sizeof(NodeGeometryMeshLine), __func__);

  node_storage->mode = GEO_NODE_MESH_LINE_MODE_DIRECTION;

  node->storage = node_storage;
}

namespace blender::nodes {

static int line_vert_total(const GeometryNodeMeshLineMode fill_type, const int verts_num)
{
  return 0;
}

static int line_edge_total(const GeometryNodeMeshLineMode fill_type, const int verts_num)
{
  return 0;
}

static Mesh *create_line_mesh(const float radius,
                              const int verts_num,
                              const GeometryNodeMeshLineMode fill_type)
{
  Mesh *mesh = BKE_mesh_new_nomain(
      line_vert_total(fill_type, verts_num), line_edge_total(fill_type, verts_num), 0, 0, 0);
  MutableSpan<MVert> verts = MutableSpan<MVert>(mesh->mvert, mesh->totvert);
  MutableSpan<MEdge> edges = MutableSpan<MEdge>(mesh->medge, mesh->totedge);

  /* Set loose edge flags. */
  for (const int i : IndexRange(verts_num)) {
    MEdge &edge = edges[i];
    edge.flag |= ME_LOOSEEDGE;
  }

  BLI_assert(BKE_mesh_is_valid(mesh));

  return mesh;
}

static void geo_node_mesh_primitive_line_exec(GeoNodeExecParams params)
{
  const NodeGeometryMeshLine &storage = *(const NodeGeometryMeshLine *)params.node().storage;
  const GeometryNodeMeshLineMode fill_type = (const GeometryNodeMeshLineMode)storage.mode;

  const int verts_num = params.extract_input<int>("Vertices");
  if (verts_num < 3) {
    params.set_output("Geometry", GeometrySet());
    return;
  }

  const float radius = params.extract_input<float>("Radius");
  const float3 location = params.extract_input<float3>("Location");
  const float3 rotation = params.extract_input<float3>("Rotation");

  Mesh *mesh = create_line_mesh(radius, verts_num, fill_type);
  BLI_assert(BKE_mesh_is_valid(mesh));

  params.set_output("Geometry", GeometrySet::create_with_mesh(mesh));
}

}  // namespace blender::nodes

void register_node_type_geo_mesh_primitive_line()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_MESH_PRIMITIVE_LINE, "Line", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(
      &ntype, geo_node_mesh_primitive_line_in, geo_node_mesh_primitive_line_out);
  node_type_init(&ntype, geo_node_mesh_primitive_line_init);
  node_type_storage(
      &ntype, "NodeGeometryMeshLine", node_free_standard_storage, node_copy_standard_storage);
  ntype.geometry_node_execute = blender::nodes::geo_node_mesh_primitive_line_exec;
  ntype.draw_buttons = geo_node_mesh_primitive_line_layout;
  nodeRegisterType(&ntype);
}
