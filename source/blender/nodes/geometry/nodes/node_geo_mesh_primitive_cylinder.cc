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
      return verts_num * 8;
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

  float angle = 0.0f;
  const float angle_delta = 2.0f * M_PI / static_cast<float>(verts_num);
  for (const int i : IndexRange(verts_num)) {
    float x = std::cos(angle) * radius;
    float y = std::sin(angle) * radius;

    copy_v3_v3(verts[i].co, float3(x, y, depth));
    copy_v3_v3(verts[verts_num + i].co, float3(x, y, -depth));
    angle += angle_delta;
  }
  if (fill_type == GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN) {
    copy_v3_v3(verts.last().co, float3(0.0f, 0.0f, depth));
    copy_v3_v3(verts[verts.size() - 2].co, float3(0.0f, 0.0f, -depth));
  }

  /* Point all vertex normals in the up direction. */
  short up_normal[3] = {0, 0, SHRT_MAX};
  short down_normal[3] = {0, 0, SHRT_MIN};
  for (const int i : IndexRange(verts_num)) {
    copy_v3_v3_short(verts[i].no, up_normal);
    copy_v3_v3_short(verts[verts_num + i].no, down_normal);
  }

  /* Create outer edges. */
  for (const int i : IndexRange(verts_num)) {
    MEdge &edge_top = edges[i];
    edge_top.v1 = i;
    edge_top.v2 = (i + 1) % verts_num;
    MEdge &edge_bottom = edges[verts_num + i];
    edge_top.v1 = verts_num + i;
    edge_top.v2 = verts_num + (i + 1) % verts_num;
  }

  /* Create triangle fan edges. */
  if (fill_type == GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN) {
    for (const int i : IndexRange(verts_num)) {
      MEdge &edge_top = edges[verts_num * 2 + i];
      edge_top.v1 = verts.size() - 1;
      edge_top.v2 = (i + 1) % verts_num;
      MEdge &edge_bottom = edges[verts_num * 3 + i];
      edge_top.v1 = verts.size() - 2;
      edge_top.v2 = verts_num + i;
    }
  }

  /* Create top corners and faces. */
  switch (fill_type) {
    case GEO_NODE_MESH_CIRCLE_FILL_NONE:
      break;
    case GEO_NODE_MESH_CIRCLE_FILL_NGON: {
      for (const int i : IndexRange(verts_num)) {
        MLoop &loop_top = loops[i];
        loop_top.e = i;
        loop_top.v = i;
        MLoop &loop_bottom = loops[verts_num + i];
        loop_bottom.e = verts_num + i;
        loop_bottom.v = verts_num + i;
      }
      MPoly &poly_top = polys[0];
      poly_top.loopstart = 0;
      poly_top.totloop = verts_num * 3;
      MPoly &poly_bottom = polys[0];
      poly_bottom.loopstart = 0;
      poly_bottom.totloop = verts_num * 3;

      break;
    }
    case GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN: {
      /* WRONG. */
      for (const int i : IndexRange(verts_num)) {
        MLoop &loop_top = loops[3 * i];
        loop_top.e = i;
        loop_top.v = i;
        MLoop &loop2_top = loops[3 * i + 1];
        loop2_top.e = verts_num * 2 + ((i + 1) % verts_num);
        loop2_top.v = (i + 1) % verts_num;
        MLoop &loop3_top = loops[3 * i + 2];
        loop3_top.e = verts_num + i;
        loop3_top.v = verts.size() - 1;

        MPoly &poly_top = polys[i];
        poly_top.loopstart = 3 * i;
        poly_top.totloop = 3;

        MLoop &loop_bottom = loops[3 * i];
        loop_bottom.e = i;
        loop_bottom.v = i;
        MLoop &loop2_bottom = loops[3 * i + 1];
        loop2_bottom.e = verts_num + ((i + 1) % verts_num);
        loop2_bottom.v = (i + 1) % verts_num;
        MLoop &loop3_bottom = loops[3 * i + 2];
        loop3_bottom.e = verts_num + i;
        loop3_bottom.v = verts.size() - 2;

        MPoly &poly_bottom = polys[i];
        poly_bottom.loopstart = 3 * i;
        poly_bottom.totloop = 3;
      }
      break;
    }
  }

  /* Create side corners and faces. */
  const int side_corner_start = verts_num * 3;
  for (const int i : IndexRange(verts_num)) {
    /* NOT DONE. */
    MLoop &loop1 = loops[side_corner_start + i];
    loop1.v = i;
    loop1.e = i;
    MLoop &loop2 = loops[side_corner_start + i];
    loop1.v = i;
    loop1.e = i;
  }

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
