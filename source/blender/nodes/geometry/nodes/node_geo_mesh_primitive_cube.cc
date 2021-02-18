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

#include "node_geometry_util.hh"

static bNodeSocketTemplate geo_node_mesh_primitive_cube_in[] = {
    {SOCK_FLOAT, N_("Size"), 0.0, 0.0, 0.0, 0.0, 0.0f, FLT_MAX, PROP_DISTANCE},
    {SOCK_VECTOR, N_("Translation"), 0.0f, 0.0f, 0.0f, 1.0f, -FLT_MAX, FLT_MAX, PROP_TRANSLATION},
    {SOCK_VECTOR, N_("Rotation"), 0.0f, 0.0f, 0.0f, 1.0f, -FLT_MAX, FLT_MAX, PROP_EULER},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_mesh_primitive_cube_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

namespace blender::nodes {

struct NewMesh {
  Mesh *mesh;
  MutableSpan<MVert> verts;
  MutableSpan<MEdge> edges;
  MutableSpan<MLoop> loops;
  MutableSpan<MPoly> polys;

  Map<std::pair<int, int>, int> edge_map;
  int edge_offset = 0;
  int loop_offset = 0;
  int poly_offset = 0;

  NewMesh(const int vert_len, const int edge_len, const int corner_len, const int poly_len)
  {
    mesh = BKE_mesh_new_nomain(vert_len, edge_len, 0, corner_len, poly_len);
    BKE_mesh_update_customdata_pointers(mesh, false);
    verts = MutableSpan<MVert>(mesh->mvert, vert_len);
    edges = MutableSpan<MEdge>(mesh->medge, edge_len);
    loops = MutableSpan<MLoop>(mesh->mloop, corner_len);
    polys = MutableSpan<MPoly>(mesh->mpoly, poly_len);

    edge_map.reserve(edge_len * 2);
  }

  int create_edge(const int vert_index_a, const int vert_index_b)
  {
    BLI_assert(vert_index_a < verts.size());
    BLI_assert(vert_index_b < verts.size());
    if (edge_map.contains({vert_index_a, vert_index_b})) {
      return edge_map.lookup({vert_index_a, vert_index_b});
    }
    if (edge_map.contains({vert_index_b, vert_index_a})) {
      return edge_map.lookup({vert_index_b, vert_index_a});
    }
    MEdge &edge = edges[edge_offset];
    edge.v1 = vert_index_a;
    edge.v2 = vert_index_b;
    edge_map.add({vert_index_a, vert_index_b}, edge_offset);
    edge_offset++;
    return edge_offset - 1;
  }

  void create_corner(const int vert_index, const int edge_index)
  {
    MLoop &loop = loops[loop_offset];
    loop.v = vert_index;
    loop.e = edge_index;
    loop_offset++;
  }

  void create_poly(const int loop_index_start, const int loops_len)
  {
    MPoly &poly = polys[poly_offset];
    poly.loopstart = loop_index_start;
    poly.totloop = loops_len;
    poly_offset++;
  }
};

static Mesh *create_cube_mesh(const float3 location, const float3 rotation, const float size)
{
  float4x4 transform;
  loc_eul_size_to_mat4(transform.values, location, rotation, float3(size));

  NewMesh new_mesh = NewMesh(8, 12, 24, 6);

  const float3 positions[8] = {
      {-1.0f, -1.0f, -1.0f},
      {-1.0f, -1.0f, 1.0f},
      {-1.0f, 1.0f, -1.0f},
      {-1.0f, 1.0f, 1.0f},
      {1.0f, -1.0f, -1.0f},
      {1.0f, -1.0f, 1.0f},
      {1.0f, 1.0f, -1.0f},
      {1.0f, 1.0f, 1.0f},
  };

  for (const int i : new_mesh.verts.index_range()) {
    MVert &vert = new_mesh.verts[i];
    float3 transformed = transform * positions[i];
    copy_v3_v3(vert.co, transformed);
  }

  Map<std::pair<int, int>, int, 12> edge_map;

  const uint8_t face_vert_indices[6][4] = {
      {0, 1, 3, 2},
      {2, 3, 7, 6},
      {6, 7, 5, 4},
      {4, 5, 1, 0},
      {2, 6, 4, 0},
      {7, 3, 1, 5},
  };

  for (const int poly_index : new_mesh.polys.index_range()) {
    for (const int corner_index : IndexRange(4)) {
      const int corner_index_next = (corner_index + 1) % 4;
      const int edge_index = new_mesh.create_edge(
          face_vert_indices[poly_index][corner_index],
          face_vert_indices[poly_index][corner_index_next]);
      new_mesh.create_corner(edge_index, face_vert_indices[poly_index][corner_index]);
    }
    new_mesh.create_poly(poly_index * 4, 4);
  }

  BLI_assert(BKE_mesh_validate(new_mesh.mesh, true, false));

  BKE_mesh_calc_normals(new_mesh.mesh);

  return new_mesh.mesh;
}

static void geo_node_mesh_primitive_cube_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set;

  const float size = params.extract_input<float>("Size");
  const float3 location = params.extract_input<float3>("Translation");
  const float3 rotation = params.extract_input<float3>("Rotation");

  geometry_set.replace_mesh(create_cube_mesh(location, rotation, size));

  params.set_output("Geometry", geometry_set);
}

}  // namespace blender::nodes

void register_node_type_geo_mesh_primitive_cube()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_MESH_PRIMITIVE_CUBE, "Cube", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(
      &ntype, geo_node_mesh_primitive_cube_in, geo_node_mesh_primitive_cube_out);
  ntype.geometry_node_execute = blender::nodes::geo_node_mesh_primitive_cube_exec;
  nodeRegisterType(&ntype);
}
