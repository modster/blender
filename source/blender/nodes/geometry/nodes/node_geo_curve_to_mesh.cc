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

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"

#include "BKE_derived_curve.hh"
#include "BKE_mesh.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "node_geometry_util.hh"

static bNodeSocketTemplate geo_node_curve_to_mesh_in[] = {
    {SOCK_GEOMETRY, N_("Curve")},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_point_translate_out[] = {
    {SOCK_GEOMETRY, N_("Mesh")},
    {-1, ""},
};

namespace blender::nodes {

static Mesh *curve_to_mesh_calculate(DCurve &curve)
{
  curve.ensure_evaluation_cache();

  if (curve.evaluated_spline_cache.size() < 1) {
    return nullptr;
  }

  Mesh *mesh = BKE_mesh_new_nomain(
      curve.evaluated_spline_cache.size(), curve.evaluated_spline_cache.size() - 2, 0, 0, 0);
  MutableSpan<MVert> verts{mesh->mvert, mesh->totvert};
  MutableSpan<MEdge> edges{mesh->medge, mesh->totedge};

  for (const int i : verts.index_range()) {
    copy_v3_v3(mesh->mvert[i].co, curve.evaluated_spline_cache[i]);
  }

  for (const int i : edges.index_range()) {
    MEdge &edge = mesh->medge[i];
    edge.v1 = i;
    edge.v2 = i + 1;
    edge.flag = ME_LOOSEEDGE;
  }

  BKE_mesh_calc_normals(mesh);
  BLI_assert(BKE_mesh_is_valid(mesh));

  return mesh;
}

static void geo_node_curve_to_mesh_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set_in = params.extract_input<GeometrySet>("Curve");
  GeometrySet geometry_set_out;

  if (geometry_set_in.has_curve()) {
    Mesh *mesh = curve_to_mesh_calculate(*geometry_set_in.get_curve_for_write());
    geometry_set_in.replace_mesh(mesh);
  }

  params.set_output("Mesh", std::move(geometry_set_out));
}

}  // namespace blender::nodes

void register_node_type_geo_curve_to_mesh()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_CURVE_TO_MESH, "Curve to Mesh", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, geo_node_curve_to_mesh_in, geo_node_point_translate_out);
  ntype.geometry_node_execute = blender::nodes::geo_node_curve_to_mesh_exec;
  nodeRegisterType(&ntype);
}
