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

#include "BKE_mesh.h"
#include "BLI_math_matrix.h"

#include "node_geometry_util.hh"

static bNodeSocketTemplate geo_node_transform_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_VECTOR, N_("Translation"), 0.0f, 0.0f, 0.0f, 1.0f, -FLT_MAX, FLT_MAX, PROP_TRANSLATION},
    {SOCK_VECTOR, N_("Rotation"), 0.0f, 0.0f, 0.0f, 1.0f, -FLT_MAX, FLT_MAX, PROP_EULER},
    {SOCK_VECTOR, N_("Scale"), 1.0f, 1.0f, 1.0f, 1.0f, -FLT_MAX, FLT_MAX, PROP_XYZ},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_transform_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

namespace blender::nodes {
static void geo_transform_exec(bNode *UNUSED(node), GValueByName &inputs, GValueByName &outputs)
{
  GeometryPtr geometry_in = inputs.extract<GeometryPtr>("Geometry");
  GeometryPtr geometry_out;

  if (!geometry_in.has_value()) {
    outputs.move_in("Geometry", std::move(geometry_out));
    return;
  }

  Mesh *mesh_in = geometry_in->mesh_get_for_read();
  if (mesh_in == nullptr) {
    outputs.move_in("Geometry", std::move(geometry_out));
    return;
  }

  const float3 translation = inputs.extract<float3>("Translation");
  const float3 rotation = inputs.extract<float3>("Rotation");
  const float3 scale = inputs.extract<float3>("Scale");

  geometry_out = GeometryPtr{new Geometry()};
  Mesh *mesh_out = BKE_mesh_copy_for_eval(mesh_in, false);

  /* Use only translation if rotation and scale are zero. */
  if (translation.length() > 0.0f && rotation.length() == 0.0f && scale.length() == 0.0f) {
    BKE_mesh_translate(mesh_out, translation, true);
  }
  else {
    float mat[4][4];
    loc_eul_size_to_mat4(mat, translation, rotation, scale);
    BKE_mesh_transform(mesh_out, mat, true);
  }

  geometry_out->mesh_set_and_transfer_ownership(mesh_out);

  outputs.move_in("Geometry", std::move(geometry_out));
}
}  // namespace blender::nodes

void register_node_type_geo_transform()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_TRANSFORM, "Transform", 0, 0);
  node_type_socket_templates(&ntype, geo_node_transform_in, geo_node_transform_out);
  ntype.geometry_node_execute = blender::nodes::geo_transform_exec;
  nodeRegisterType(&ntype);
}
