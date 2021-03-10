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

static bNodeSocketTemplate geo_node_mesh_primitive_cylinder_in[] = {
    {SOCK_INT, N_("Vertices"), 32, 0.0f, 0.0f, 0.0f, 3, 4096},
    {SOCK_FLOAT, N_("Radius"), 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, FLT_MAX, PROP_DISTANCE},
    {SOCK_FLOAT, N_("Depth"), 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, FLT_MAX, PROP_DISTANCE},
    {SOCK_VECTOR, N_("Location"), 0.0f, 0.0f, 0.0f, 1.0f, -FLT_MAX, FLT_MAX, PROP_TRANSLATION},
    {SOCK_VECTOR, N_("Rotation"), 0.0f, 0.0f, 0.0f, 1.0f, -FLT_MAX, FLT_MAX, PROP_EULER},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_mesh_primitive_cylinder_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

static void geo_node_mesh_primitive_cylinder_layout(uiLayout *layout,
                                                    bContext *UNUSED(C),
                                                    PointerRNA *ptr)
{
  uiLayoutSetPropSep(layout, true);
  uiLayoutSetPropDecorate(layout, false);
  uiItemR(layout, ptr, "fill_type", 0, nullptr, ICON_NONE);
}

static void geo_node_mesh_primitive_cylinder_init(bNodeTree *UNUSED(ntree), bNode *node)
{
  NodeGeometryMeshCylinder *node_storage = (NodeGeometryMeshCylinder *)MEM_callocN(
      sizeof(NodeGeometryMeshCylinder), __func__);

  node_storage->fill_type = GEO_NODE_MESH_CIRCLE_FILL_NONE;

  node->storage = node_storage;
}

namespace blender::nodes {

static int vert_total(const GeometryNodeMeshCircleFillType fill_type, const int verts_num)
{
  switch (fill_type) {
    case GEO_NODE_MESH_CIRCLE_FILL_NONE:
    case GEO_NODE_MESH_CIRCLE_FILL_NGON:
      return verts_num * 2;
    case GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN:
      return verts_num * 2 + 2;
  }
  BLI_assert(false);
  return 0;
}

static int edge_total(const GeometryNodeMeshCircleFillType fill_type, const int verts_num)
{
  switch (fill_type) {
    case GEO_NODE_MESH_CIRCLE_FILL_NONE:
    case GEO_NODE_MESH_CIRCLE_FILL_NGON:
      return verts_num * 3;
    case GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN:
      return verts_num * 5;
  }
  BLI_assert(false);
  return 0;
}

static int corner_total(const GeometryNodeMeshCircleFillType fill_type, const int verts_num)
{
  switch (fill_type) {
    case GEO_NODE_MESH_CIRCLE_FILL_NONE:
      return verts_num * 4;
    case GEO_NODE_MESH_CIRCLE_FILL_NGON:
      return verts_num * 6;
    case GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN:
      return verts_num * 10;
  }
  BLI_assert(false);
  return 0;
}

static int face_total(const GeometryNodeMeshCircleFillType fill_type, const int verts_num)
{
  switch (fill_type) {
    case GEO_NODE_MESH_CIRCLE_FILL_NONE:
      return verts_num;
    case GEO_NODE_MESH_CIRCLE_FILL_NGON:
      return verts_num + 2;
    case GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN:
      return verts_num * 3;
  }
  BLI_assert(false);
  return 0;
}

static Mesh *create_cylinder_mesh(const float radius,
                                  const float depth,
                                  const int verts_num,
                                  const GeometryNodeMeshCircleFillType fill_type)
{
  Mesh *mesh = BKE_mesh_new_nomain(vert_total(fill_type, verts_num),
                                   edge_total(fill_type, verts_num),
                                   0,
                                   corner_total(fill_type, verts_num),
                                   face_total(fill_type, verts_num));
  MutableSpan<MVert> verts = MutableSpan<MVert>(mesh->mvert, mesh->totvert);
  MutableSpan<MEdge> edges = MutableSpan<MEdge>(mesh->medge, mesh->totedge);
  MutableSpan<MLoop> loops = MutableSpan<MLoop>(mesh->mloop, mesh->totloop);
  MutableSpan<MPoly> polys = MutableSpan<MPoly>(mesh->mpoly, mesh->totpoly);

  /* Calculate vertex data. */
  const int top_verts_start = 0;
  const int bottom_verts_start = verts_num;
  float angle = 0.0f;
  const float angle_delta = 2.0f * M_PI / static_cast<float>(verts_num);
  for (const int i : IndexRange(verts_num)) {
    float x = std::cos(angle) * radius;
    float y = std::sin(angle) * radius;

    copy_v3_v3(verts[top_verts_start + i].co, float3(x, y, depth));
    copy_v3_v3(verts[bottom_verts_start + i].co, float3(x, y, -depth));
    angle += angle_delta;
  }

  /* Add center vertices for the triangle fans. */
  const int top_center_vert_index = verts.size() - 1;
  const int bottom_center_vert_index = verts.size() - 2;
  if (fill_type == GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN) {
    copy_v3_v3(verts[top_center_vert_index].co, float3(0.0f, 0.0f, depth));
    copy_v3_v3(verts[bottom_center_vert_index].co, float3(0.0f, 0.0f, -depth));
  }

  /* Create outer edges. */
  const int top_edges_start = 0;
  const int connecting_edges_start = verts_num;
  const int bottom_edges_start = verts_num * 2;
  for (const int i : IndexRange(verts_num)) {
    MEdge &edge_top = edges[top_edges_start + i];
    edge_top.v1 = top_verts_start + i;
    edge_top.v2 = top_verts_start + (i + 1) % verts_num;
    MEdge &edge_connecting = edges[connecting_edges_start + i];
    edge_connecting.v1 = top_verts_start + i;
    edge_connecting.v2 = bottom_verts_start + i;
    MEdge &edge_bottom = edges[bottom_edges_start + i];
    edge_bottom.v1 = bottom_verts_start + i;
    edge_bottom.v2 = bottom_verts_start + (i + 1) % verts_num;
  }

  /* Create triangle fan edges. */
  const int top_fan_edges_start = verts_num * 3;
  const int bottom_fan_edges_start = verts_num * 4;
  if (fill_type == GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN) {
    for (const int i : IndexRange(verts_num)) {
      MEdge &edge_top = edges[top_fan_edges_start + i];
      edge_top.v1 = top_center_vert_index;
      edge_top.v2 = top_verts_start + i;
      MEdge &edge_bottom = edges[bottom_fan_edges_start + i];
      edge_bottom.v1 = bottom_center_vert_index;
      edge_bottom.v2 = bottom_verts_start + i;
    }
  }

  /* Create top corners and faces. */
  int loop_index = 0;
  int poly_index = 0;
  switch (fill_type) {
    case GEO_NODE_MESH_CIRCLE_FILL_NONE:
      break;
    case GEO_NODE_MESH_CIRCLE_FILL_NGON: {
      MPoly &poly = polys[poly_index++];
      poly.loopstart = loop_index;
      poly.totloop = verts_num;

      for (const int i : IndexRange(verts_num)) {
        MLoop &loop = loops[loop_index++];
        loop.v = top_verts_start + i;
        loop.e = top_edges_start + i;
      }
      break;
    }
    case GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN: {
      for (const int i : IndexRange(verts_num)) {
        MPoly &poly = polys[poly_index++];
        poly.loopstart = loop_index;
        poly.totloop = 3;

        MLoop &loop1 = loops[loop_index++];
        loop1.v = top_verts_start + i;
        loop1.e = top_edges_start + i;
        MLoop &loop2 = loops[loop_index++];
        loop2.v = top_verts_start + (i + 1) % verts_num;
        loop2.e = top_fan_edges_start + (i + 1) % verts_num;
        MLoop &loop3 = loops[loop_index++];
        loop3.v = top_center_vert_index;
        loop3.e = top_fan_edges_start + i;
      }
      break;
    }
  }

  /* Create side corners and faces. */
  for (const int i : IndexRange(verts_num)) {
    MPoly &poly = polys[poly_index++];
    poly.loopstart = loop_index;
    poly.totloop = 4;

    MLoop &loop1 = loops[loop_index++];
    loop1.v = top_verts_start + i;
    loop1.e = connecting_edges_start + i;
    MLoop &loop2 = loops[loop_index++];
    loop2.v = bottom_verts_start + i;
    loop2.e = bottom_edges_start + i;
    MLoop &loop3 = loops[loop_index++];
    loop3.v = bottom_verts_start + (i + 1) % verts_num;
    loop3.e = connecting_edges_start + (i + 1) % verts_num;
    MLoop &loop4 = loops[loop_index++];
    loop4.v = top_verts_start + (i + 1) % verts_num;
    loop4.e = top_edges_start + i;
  }

  /* Create bottom corners and faces. */
  switch (fill_type) {
    case GEO_NODE_MESH_CIRCLE_FILL_NONE:
      break;
    case GEO_NODE_MESH_CIRCLE_FILL_NGON: {
      MPoly &poly = polys[poly_index++];
      poly.loopstart = loop_index;
      poly.totloop = verts_num;

      for (const int i : IndexRange(verts_num)) {
        MLoop &loop = loops[loop_index++];
        loop.e = bottom_edges_start + i;
        loop.v = bottom_verts_start + i;
      }
      break;
    }
    case GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN: {
      for (const int i : IndexRange(verts_num)) {
        MPoly &poly = polys[poly_index++];
        poly.loopstart = loop_index;
        poly.totloop = 3;

        MLoop &loop1 = loops[loop_index++];
        loop1.v = bottom_verts_start + i;
        loop1.e = bottom_fan_edges_start + i;
        MLoop &loop2 = loops[loop_index++];
        loop2.v = bottom_center_vert_index;
        loop2.e = bottom_fan_edges_start + (i + 1) % verts_num;
        MLoop &loop3 = loops[loop_index++];
        loop3.v = bottom_verts_start + (i + 1) % verts_num;
        loop3.e = bottom_edges_start + i;
      }
      break;
    }
  }

  BKE_mesh_calc_normals(mesh);

  return mesh;
}

static void geo_node_mesh_primitive_cylinder_exec(GeoNodeExecParams params)
{
  const bNode &node = params.node();
  const NodeGeometryMeshCylinder &storage = *(const NodeGeometryMeshCylinder *)node.storage;

  const GeometryNodeMeshCircleFillType fill_type = (const GeometryNodeMeshCircleFillType)
                                                       storage.fill_type;

  const int verts_num = params.extract_input<int>("Vertices");
  if (verts_num < 3) {
    params.set_output("Geometry", GeometrySet());
    return;
  }

  const float radius = params.extract_input<float>("Radius");
  const float depth = params.extract_input<float>("Depth");
  const float3 location = params.extract_input<float3>("Location");
  const float3 rotation = params.extract_input<float3>("Rotation");

  Mesh *mesh = create_cylinder_mesh(radius, depth, verts_num, fill_type);

  if (!location.is_zero() || !rotation.is_zero()) {
    transform_mesh(mesh, location, rotation, float3(1));
  }

  BLI_assert(BKE_mesh_is_valid(mesh));

  params.set_output("Geometry", GeometrySet::create_with_mesh(mesh));
}

}  // namespace blender::nodes

void register_node_type_geo_mesh_primitive_cylinder()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_MESH_PRIMITIVE_CYLINDER, "Cylinder", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(
      &ntype, geo_node_mesh_primitive_cylinder_in, geo_node_mesh_primitive_cylinder_out);
  node_type_init(&ntype, geo_node_mesh_primitive_cylinder_init);
  node_type_storage(
      &ntype, "NodeGeometryMeshCylinder", node_free_standard_storage, node_copy_standard_storage);
  ntype.geometry_node_execute = blender::nodes::geo_node_mesh_primitive_cylinder_exec;
  ntype.draw_buttons = geo_node_mesh_primitive_cylinder_layout;
  nodeRegisterType(&ntype);
}
