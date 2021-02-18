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

static bNodeSocketTemplate geo_node_mesh_primitive_circle_in[] = {
    {SOCK_INT, N_("Vertices"), 16, 0.0f, 0.0f, 0.0f, 3, 4096},
    {SOCK_FLOAT, N_("Radius"), 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, FLT_MAX, PROP_DISTANCE},
    {SOCK_VECTOR, N_("Location"), 0.0f, 0.0f, 0.0f, 1.0f, -FLT_MAX, FLT_MAX, PROP_TRANSLATION},
    {SOCK_VECTOR, N_("Rotation"), 0.0f, 0.0f, 0.0f, 1.0f, -FLT_MAX, FLT_MAX, PROP_EULER},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_mesh_primitive_circle_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

static void geo_node_mesh_primitive_circle_layout(uiLayout *layout,
                                                  bContext *UNUSED(C),
                                                  PointerRNA *ptr)
{
  uiItemR(layout, ptr, "fill_type", 0, nullptr, ICON_NONE);
}

static void geo_node_mesh_primitive_circle_init(bNodeTree *UNUSED(ntree), bNode *node)
{
  NodeGeometryMeshCircle *node_storage = (NodeGeometryMeshCircle *)MEM_callocN(
      sizeof(NodeGeometryMeshCircle), __func__);

  node_storage->fill_type = GEO_NODE_MESH_CIRCLE_FILL_NONE;

  node->storage = node_storage;
}

namespace blender::nodes {

static int vert_total(const GeometryNodeMeshCircleFillType fill_type, const int verts_num)
{
  switch (fill_type) {
    case GEO_NODE_MESH_CIRCLE_FILL_NONE:
    case GEO_NODE_MESH_CIRCLE_FILL_NGON:
      return verts_num;
    case GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN:
      return verts_num + 1;
  }
  BLI_assert(false);
  return 0;
}

static int edge_total(const GeometryNodeMeshCircleFillType fill_type, const int verts_num)
{
  switch (fill_type) {
    case GEO_NODE_MESH_CIRCLE_FILL_NONE:
    case GEO_NODE_MESH_CIRCLE_FILL_NGON:
      return verts_num;
    case GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN:
      return verts_num * 2;
  }
  BLI_assert(false);
  return 0;
}

static int corner_total(const GeometryNodeMeshCircleFillType fill_type, const int verts_num)
{
  switch (fill_type) {
    case GEO_NODE_MESH_CIRCLE_FILL_NONE:
      return 0;
    case GEO_NODE_MESH_CIRCLE_FILL_NGON:
      return verts_num;
    case GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN:
      return 3 * verts_num;
  }
  BLI_assert(false);
  return 0;
}

static int face_total(const GeometryNodeMeshCircleFillType fill_type, const int verts_num)
{
  switch (fill_type) {
    case GEO_NODE_MESH_CIRCLE_FILL_NONE:
      return 0;
    case GEO_NODE_MESH_CIRCLE_FILL_NGON:
      return 1;
    case GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN:
      return verts_num;
  }
  BLI_assert(false);
  return 0;
}

static Mesh *create_circle_mesh(const float3 location,
                                const float3 rotation,
                                const float radius,
                                const int verts_num,
                                const GeometryNodeMeshCircleFillType fill_type)
{
  float4x4 transform;
  loc_eul_size_to_mat4(transform.values, location, rotation, float3(1.0f));

  Mesh *mesh = BKE_mesh_new_nomain(vert_total(fill_type, verts_num),
                                   edge_total(fill_type, verts_num),
                                   0,
                                   corner_total(fill_type, verts_num),
                                   face_total(fill_type, verts_num));
  MutableSpan<MVert> verts = MutableSpan<MVert>(mesh->mvert, mesh->totvert);
  MutableSpan<MEdge> edges = MutableSpan<MEdge>(mesh->medge, mesh->totedge);
  MutableSpan<MLoop> loops = MutableSpan<MLoop>(mesh->mloop, mesh->totloop);
  MutableSpan<MPoly> polys = MutableSpan<MPoly>(mesh->mpoly, mesh->totpoly);

  {
    float angle = 0.0f;
    const float angle_delta = 2.0f * M_PI / static_cast<float>(verts_num);
    for (const int i : IndexRange(verts_num)) {
      MVert &vert = verts[i];
      float3 co = float3(std::cos(angle) * radius, std::sin(angle) * radius, 0.0f) + location;

      copy_v3_v3(vert.co, co);
      angle += angle_delta;
    }
    if (fill_type == GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN) {
      copy_v3_v3(verts.last().co, float3(0.0f, 0.0f, 0.0f));
    }
  }

  /* Create outer edges. */
  for (const int i : IndexRange(verts_num - 1)) {
    MEdge &edge = edges[i];
    edge.v1 = i;
    edge.v2 = i + 1;
  }
  edges.last().v1 = verts_num - 1;
  edges.last().v2 = 0;

  if (fill_type == GEO_NODE_MESH_CIRCLE_FILL_NONE) {
    for (MEdge &edge : edges) {
      edge.flag |= ME_LOOSEEDGE;
    }
  }

  /* Create triangle fan edges. */
  if (fill_type == GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN) {
  }

  /* Create corners. */
  switch (fill_type) {
    case GEO_NODE_MESH_CIRCLE_FILL_NONE:
      break;
    case GEO_NODE_MESH_CIRCLE_FILL_NGON: {
      for (const int i : IndexRange(verts_num)) {
        MLoop &loop = loops[i];
        loop.e = i;
        loop.v = i;
      }
      break;
    }
    case GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN: {
      for (const int i : IndexRange(verts_num)) {
        MLoop &loop = loops[3 * i];
        loop.e = i;
        loop.v = i;
        MLoop &loop2 = loops[3 * i + 1];
        loop2.e = verts_num + 1;
        loop2.v = i + i;
      }
      break;
    }
  }

  /* Create face(s). */
  switch (fill_type) {
    case GEO_NODE_MESH_CIRCLE_FILL_NONE:
      break;
    case GEO_NODE_MESH_CIRCLE_FILL_NGON: {
      MPoly &poly = polys[0];
      poly.loopstart = 0;
      poly.totloop = loops.size();
      break;
    }
    case GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN: {
      break;
    }
  }

  BLI_assert(BKE_mesh_validate(mesh, true, false));

  return mesh;
}

static void geo_node_mesh_primitive_circle_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set;

  const bNode &node = params.node();
  const NodeGeometryMeshCircle &storage = *(const NodeGeometryMeshCircle *)node.storage;

  const GeometryNodeMeshCircleFillType fill_type = (const GeometryNodeMeshCircleFillType)
                                                       storage.fill_type;

  const int verts_num = params.extract_input<int>("Vertices");
  if (verts_num < 3) {
    params.set_output("Geometry", geometry_set);
    return;
  }

  const float radius = params.extract_input<float>("Radius");
  const float3 location = params.extract_input<float3>("Location");
  const float3 rotation = params.extract_input<float3>("Rotation");

  geometry_set.replace_mesh(create_circle_mesh(location, rotation, radius, verts_num, fill_type));

  params.set_output("Geometry", geometry_set);
}

}  // namespace blender::nodes

void register_node_type_geo_mesh_primitive_circle()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_MESH_PRIMITIVE_CIRCLE, "Circle", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(
      &ntype, geo_node_mesh_primitive_circle_in, geo_node_mesh_primitive_circle_out);
  node_type_init(&ntype, geo_node_mesh_primitive_circle_init);
  node_type_storage(
      &ntype, "NodeGeometryMeshCircle", node_free_standard_storage, node_copy_standard_storage);
  ntype.geometry_node_execute = blender::nodes::geo_node_mesh_primitive_circle_exec;
  ntype.draw_buttons = geo_node_mesh_primitive_circle_layout;
  nodeRegisterType(&ntype);
}
