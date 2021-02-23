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

static bNodeSocketTemplate geo_node_mesh_primitive_uv_sphere_in[] = {
    {SOCK_INT, N_("Segments"), 32, 0.0f, 0.0f, 0.0f, 3, 1024},
    {SOCK_INT, N_("Rings"), 16, 0.0f, 0.0f, 0.0f, 3, 1024},
    {SOCK_FLOAT, N_("Radius"), 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, FLT_MAX, PROP_DISTANCE},
    {SOCK_VECTOR, N_("Location"), 0.0f, 0.0f, 0.0f, 1.0f, -FLT_MAX, FLT_MAX, PROP_TRANSLATION},
    {SOCK_VECTOR, N_("Rotation"), 0.0f, 0.0f, 0.0f, 1.0f, -FLT_MAX, FLT_MAX, PROP_EULER},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_mesh_primitive_uv_sphere_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

namespace blender::nodes {

static int vert_total(const int segments, const int rings)
{
  return 0;
}

static int edge_total(const int segments, const int rings)
{
  return 0;
}

static int corner_total(const int segments, const int rings)
{
  return 0;
}

static int face_total(const int segments, const int rings)
{
  return 0;
}

static Mesh *create_uv_sphere_mesh(const float3 location,
                                   const float3 rotation,
                                   const float radius,
                                   const int segments,
                                   const int rings)
{
  float4x4 transform;
  loc_eul_size_to_mat4(transform.values, location, rotation, float3(1.0f));

  Mesh *mesh = BKE_mesh_new_nomain(vert_total(segments, rings),
                                   edge_total(segments, rings),
                                   0,
                                   corner_total(segments, rings),
                                   face_total(segments, rings));
  MutableSpan<MVert> verts = MutableSpan<MVert>(mesh->mvert, mesh->totvert);
  MutableSpan<MEdge> edges = MutableSpan<MEdge>(mesh->medge, mesh->totedge);
  MutableSpan<MLoop> loops = MutableSpan<MLoop>(mesh->mloop, mesh->totloop);
  MutableSpan<MPoly> polys = MutableSpan<MPoly>(mesh->mpoly, mesh->totpoly);

  BLI_assert(BKE_mesh_is_valid(mesh));

  return mesh;
}

static void geo_node_mesh_primitive_uv_sphere_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set;

  const bNode &node = params.node();

  const int segments_num = params.extract_input<int>("Vertices");
  const int rings_num = params.extract_input<int>("Vertices");
  if (segments_num < 3 || rings_num < 3) {
    params.set_output("Geometry", geometry_set);
    return;
  }

  const float radius = params.extract_input<float>("Radius");
  const float3 location = params.extract_input<float3>("Location");
  const float3 rotation = params.extract_input<float3>("Rotation");

  if (rotation.length_squared() == 0.0f) {
    params.error_message_add(NodeWarningType::Warning, "Rotation is zero");
  }
  const float3 rotation_normalized = rotation.length_squared() == 0.0f ? float3(0.0f, 0.0f, 1.0f) :
                                                                         rotation.normalized();

  geometry_set.replace_mesh(
      create_uv_sphere_mesh(location, rotation_normalized, radius, segments_num, rings_num));

  params.set_output("Geometry", geometry_set);
}

}  // namespace blender::nodes

void register_node_type_geo_mesh_primitive_uv_sphere()
{
  static bNodeType ntype;

  geo_node_type_base(
      &ntype, GEO_NODE_MESH_PRIMITIVE_UV_SPHERE, "UV Sphere", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(
      &ntype, geo_node_mesh_primitive_uv_sphere_in, geo_node_mesh_primitive_uv_sphere_out);
  ntype.geometry_node_execute = blender::nodes::geo_node_mesh_primitive_uv_sphere_exec;
  nodeRegisterType(&ntype);
}
