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

static int vert_total(const GeometryNodeMeshCircleFillType fill_type,
                      const int verts_num,
                      const bool use_top,
                      const bool use_bottom)
{
  int vert_total = 0;
  if (use_top) {
    vert_total += verts_num;
    if (fill_type == GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN) {
      vert_total++;
    }
  }
  else {
    vert_total++;
  }
  if (use_bottom) {
    vert_total += verts_num;
    if (fill_type == GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN) {
      vert_total++;
    }
  }
  else {
    vert_total++;
  }

  return vert_total;
}

static int edge_total(const GeometryNodeMeshCircleFillType fill_type,
                      const int verts_num,
                      const bool use_top,
                      const bool use_bottom)
{
  if (!use_top && !use_bottom) {
    return 1;
  }

  int edge_total = 0;
  if (use_top) {
    edge_total += verts_num;
    if (fill_type == GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN) {
      edge_total += verts_num;
    }
  }

  edge_total += verts_num;

  if (use_bottom) {
    edge_total += verts_num;
    if (fill_type == GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN) {
      edge_total += verts_num;
    }
  }

  return edge_total;
}

static int corner_total(const GeometryNodeMeshCircleFillType fill_type,
                        const int verts_num,
                        const bool use_top,
                        const bool use_bottom)
{
  if (!use_top && !use_bottom) {
    return 0;
  }

  int corner_total = 0;
  if (use_top) {
    if (fill_type == GEO_NODE_MESH_CIRCLE_FILL_NGON) {
      corner_total += verts_num;
    }
    else if (fill_type == GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN) {
      corner_total += verts_num * 3;
    }
  }

  if (use_top && use_bottom) {
    corner_total += verts_num * 4;
  }
  else {
    corner_total += verts_num * 3;
  }

  if (use_bottom) {
    if (fill_type == GEO_NODE_MESH_CIRCLE_FILL_NGON) {
      corner_total += verts_num;
    }
    else if (fill_type == GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN) {
      corner_total += verts_num * 3;
    }
  }

  return corner_total;
}

static int face_total(const GeometryNodeMeshCircleFillType fill_type,
                      const int verts_num,
                      const bool use_top,
                      const bool use_bottom)
{
  if (!use_top && !use_bottom) {
    return 0;
  }

  int face_total = 0;
  if (use_top) {
    if (fill_type == GEO_NODE_MESH_CIRCLE_FILL_NGON) {
      face_total++;
    }
    else if (fill_type == GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN) {
      face_total += verts_num;
    }
  }

  face_total += verts_num;

  if (use_bottom) {
    if (fill_type == GEO_NODE_MESH_CIRCLE_FILL_NGON) {
      face_total++;
    }
    else if (fill_type == GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN) {
      face_total += verts_num;
    }
  }

  return face_total;
}

Mesh *create_cylinder_or_cone_mesh(const float radius_top,
                                   const float radius_bottom,
                                   const float depth,
                                   const int verts_num,
                                   const GeometryNodeMeshCircleFillType fill_type)
{
  const bool use_top = radius_top != 0.0f;
  const bool use_bottom = radius_bottom != 0.0f;
  /* Handle the case of a line / single point before everything else to avoid
   * the need to check for it later. */
  if (!use_top && !use_bottom) {
    const bool single_vertex = depth == 0.0f;
    Mesh *mesh = BKE_mesh_new_nomain(single_vertex ? 1 : 2, single_vertex ? 0 : 1, 0, 0, 0);
    copy_v3_v3(mesh->mvert[0].co, float3(0.0f, 0.0f, depth));
    if (single_vertex) {
      short up[3] = {0, 0, SHRT_MAX};
      copy_v3_v3_short(mesh->mvert[0].no, up);
      return mesh;
    }
    copy_v3_v3(mesh->mvert[1].co, float3(0.0f, 0.0f, -depth));
    mesh->medge[0].v1 = 0;
    mesh->medge[0].v2 = 1;
    mesh->medge[0].flag |= ME_LOOSEEDGE;
    BKE_mesh_calc_normals(mesh);
    return mesh;
  }

  Mesh *mesh = BKE_mesh_new_nomain(vert_total(fill_type, verts_num, use_top, use_bottom),
                                   edge_total(fill_type, verts_num, use_top, use_bottom),
                                   0,
                                   corner_total(fill_type, verts_num, use_top, use_bottom),
                                   face_total(fill_type, verts_num, use_top, use_bottom));
  MutableSpan<MVert> verts = MutableSpan<MVert>(mesh->mvert, mesh->totvert);
  MutableSpan<MEdge> edges = MutableSpan<MEdge>(mesh->medge, mesh->totedge);
  MutableSpan<MLoop> loops = MutableSpan<MLoop>(mesh->mloop, mesh->totloop);
  MutableSpan<MPoly> polys = MutableSpan<MPoly>(mesh->mpoly, mesh->totpoly);

  /* Calculate vertex positions. */
  const int top_verts_start = 0;
  const int bottom_verts_start = top_verts_start + (use_top ? verts_num : 1);
  float angle = 0.0f;
  const float angle_delta = 2.0f * M_PI / static_cast<float>(verts_num);
  for (const int i : IndexRange(verts_num)) {
    const float x = std::cos(angle);
    const float y = std::sin(angle);
    if (use_top) {
      copy_v3_v3(verts[top_verts_start + i].co, float3(x * radius_top, y * radius_top, depth));
    }
    if (use_bottom) {
      copy_v3_v3(verts[bottom_verts_start + i].co,
                 float3(x * radius_bottom, y * radius_bottom, -depth));
    }
    angle += angle_delta;
  }
  if (!use_top) {
    copy_v3_v3(verts[top_verts_start].co, float3(0.0f, 0.0f, depth));
  }
  if (!use_bottom) {
    copy_v3_v3(verts[bottom_verts_start].co, float3(0.0f, 0.0f, -depth));
  }

  /* Add center vertices for the triangle fans at the end. */
  const int top_center_vert_index = bottom_verts_start + (use_bottom ? verts_num : 1);
  const int bottom_center_vert_index = top_center_vert_index + (use_top ? 1 : 0);
  if (fill_type == GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN) {
    if (use_top) {
      copy_v3_v3(verts[top_center_vert_index].co, float3(0.0f, 0.0f, depth));
    }
    if (use_bottom) {
      copy_v3_v3(verts[bottom_center_vert_index].co, float3(0.0f, 0.0f, -depth));
    }
  }

  /* Create top edges. */
  const int top_edges_start = 0;
  const int top_fan_edges_start = (use_top &&
                                   fill_type == GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN) ?
                                      top_edges_start + verts_num :
                                      top_edges_start;
  if (use_top) {
    for (const int i : IndexRange(verts_num)) {
      MEdge &edge = edges[top_edges_start + i];
      edge.v1 = top_verts_start + i;
      edge.v2 = top_verts_start + (i + 1) % verts_num;
    }
    if (fill_type == GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN) {
      for (const int i : IndexRange(verts_num)) {
        MEdge &edge = edges[top_fan_edges_start + i];
        edge.v1 = top_center_vert_index;
        edge.v2 = top_verts_start + i;
      }
    }
  }

  /* Create connecting edges. */
  const int connecting_edges_start = top_fan_edges_start + (use_top ? verts_num : 0);
  for (const int i : IndexRange(verts_num)) {
    MEdge &edge = edges[connecting_edges_start + i];
    edge.v1 = top_verts_start + (use_top ? i : 0);
    edge.v2 = bottom_verts_start + (use_bottom ? i : 0);
  }

  /* Create bottom edges. */
  const int bottom_edges_start = connecting_edges_start + verts_num;
  const int bottom_fan_edges_start = (use_bottom &&
                                      fill_type == GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN) ?
                                         bottom_edges_start + verts_num :
                                         bottom_edges_start;
  if (use_bottom) {
    for (const int i : IndexRange(verts_num)) {
      MEdge &edge = edges[bottom_edges_start + i];
      edge.v1 = bottom_verts_start + i;
      edge.v2 = bottom_verts_start + (i + 1) % verts_num;
    }
    if (fill_type == GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN) {
      for (const int i : IndexRange(verts_num)) {
        MEdge &edge = edges[bottom_fan_edges_start + i];
        edge.v1 = bottom_center_vert_index;
        edge.v2 = bottom_verts_start + i;
      }
    }
  }

  /* Create top corners and faces. */
  int loop_index = 0;
  int poly_index = 0;
  if (use_top) {
    if (fill_type == GEO_NODE_MESH_CIRCLE_FILL_NGON) {
      MPoly &poly = polys[poly_index++];
      poly.loopstart = loop_index;
      poly.totloop = verts_num;

      for (const int i : IndexRange(verts_num)) {
        MLoop &loop = loops[loop_index++];
        loop.v = top_verts_start + i;
        loop.e = top_edges_start + i;
      }
    }
    else if (fill_type == GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN) {
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
    }
  }

  /* Create side corners and faces. */
  if (use_top && use_bottom) {
    /* Quads connect the top and bottom. */
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
  }
  else {
    /* Triangles connect the top and bottom section. */
    if (use_top) {
      for (const int i : IndexRange(verts_num)) {
        MPoly &poly = polys[poly_index++];
        poly.loopstart = loop_index;
        poly.totloop = 3;

        MLoop &loop1 = loops[loop_index++];
        loop1.v = top_verts_start + i;
        loop1.e = connecting_edges_start + i;
        MLoop &loop2 = loops[loop_index++];
        loop2.v = bottom_verts_start;
        loop2.e = connecting_edges_start + (i + 1) % verts_num;
        MLoop &loop3 = loops[loop_index++];
        loop3.v = top_verts_start + (i + 1) % verts_num;
        loop3.e = top_edges_start + i;
      }
    }
    else {
      BLI_assert(use_bottom);
      for (const int i : IndexRange(verts_num)) {
        MPoly &poly = polys[poly_index++];
        poly.loopstart = loop_index;
        poly.totloop = 3;

        MLoop &loop1 = loops[loop_index++];
        loop1.v = bottom_verts_start + i;
        loop1.e = bottom_edges_start + i;
        MLoop &loop2 = loops[loop_index++];
        loop2.v = bottom_verts_start + (i + 1) % verts_num;
        loop2.e = connecting_edges_start + (i + 1) % verts_num;
        MLoop &loop3 = loops[loop_index++];
        loop3.v = top_verts_start;
        loop3.e = connecting_edges_start + i;
      }
    }
  }

  /* Create bottom corners and faces. */
  if (use_bottom) {
    if (fill_type == GEO_NODE_MESH_CIRCLE_FILL_NGON) {
      MPoly &poly = polys[poly_index++];
      poly.loopstart = loop_index;
      poly.totloop = verts_num;

      for (const int i : IndexRange(verts_num)) {
        /* Go backwards to reverse surface normal. */
        MLoop &loop = loops[loop_index++];
        loop.v = bottom_verts_start + verts_num - 1 - i;
        loop.e = bottom_edges_start + verts_num - 1 - (i + 1) % verts_num;
      }
    }
    else if (fill_type == GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN) {
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

  Mesh *mesh = create_cylinder_or_cone_mesh(radius, radius, depth, verts_num, fill_type);

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
