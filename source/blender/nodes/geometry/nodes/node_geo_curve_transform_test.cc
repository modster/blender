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

#include "BLI_float4x4.hh"

#include "BKE_mesh.h"

#include "node_geometry_util.hh"

static bNodeSocketTemplate geo_node_transform_test_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_VECTOR, N_("Translation"), 0.0f, 0.0f, 0.0f, 1.0f, -FLT_MAX, FLT_MAX, PROP_TRANSLATION},
    {SOCK_VECTOR, N_("Forward"), 0.0f, 0.0f, 0.0f, 1.0f, -FLT_MAX, FLT_MAX},
    {SOCK_VECTOR, N_("Up"), 0.0f, 0.0f, 0.0f, 1.0f, -FLT_MAX, FLT_MAX},
    {SOCK_FLOAT, N_("Radius"), 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, FLT_MAX},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_point_translate_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

namespace blender::nodes {

static void geo_node_transform_test_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");
  const float3 translation = params.extract_input<float3>("Translation");
  const float3 forward = params.extract_input<float3>("Forward").normalized();
  const float3 up = params.extract_input<float3>("Up").normalized();
  const float radius = params.extract_input<float>("Radius");

  float4x4 matrix = float4x4::from_normalized_axis_data(translation, forward, up);
  // float4x4 scale_matrix;
  // scale_m4_fl(scale_matrix.values, radius * 2.0f);
  // const float4x4 final_matrix = matrix * scale_matrix;
  matrix.apply_scale(radius);
  // const float4x4 final_matrix = matrix;

  if (geometry_set.has_mesh()) {
    Mesh *mesh = geometry_set.get_mesh_for_write();
    BKE_mesh_transform(mesh, matrix.values, false);
  }

  params.set_output("Geometry", geometry_set);
}

}  // namespace blender::nodes

void register_node_type_geo_curve_transform_test()
{
  static bNodeType ntype;

  geo_node_type_base(
      &ntype, GEO_NODE_CURVE_TRANSFORM_TEST, "Transform Test", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, geo_node_transform_test_in, geo_node_point_translate_out);
  ntype.geometry_node_execute = blender::nodes::geo_node_transform_test_exec;
  nodeRegisterType(&ntype);
}
