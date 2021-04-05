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
    {SOCK_GEOMETRY, N_("Profile Curve")},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_point_translate_out[] = {
    {SOCK_GEOMETRY, N_("Mesh")},
    {-1, ""},
};

namespace blender::nodes {

static void vert_extrude_to_mesh_data(const Spline &spline,
                                      const float3 profile_vert,
                                      MutableSpan<MVert> verts,
                                      MutableSpan<MEdge> edges,
                                      int &vert_offset,
                                      int &edge_offset)
{
  Span<float3> positions = spline.evaluated_positions();

  for (const int i : IndexRange(positions.size() - 1)) {
    MEdge &edge = edges[edge_offset++];
    edge.v1 = vert_offset + i;
    edge.v2 = vert_offset + i + 1;
    edge.flag = ME_LOOSEEDGE;
  }

  if (spline.is_cyclic) {
    MEdge &edge = edges[edge_offset++];
    edge.v1 = vert_offset;
    edge.v2 = vert_offset + positions.size() - 1;
    edge.flag = ME_LOOSEEDGE;
  }

  for (const int i : positions.index_range()) {
    MVert &vert = verts[vert_offset++];
    copy_v3_v3(vert.co, positions[i] + profile_vert);
  }
}

static void spline_extrude_to_mesh_data(const Spline &spline,
                                        const Spline &profile_spline,
                                        MutableSpan<MVert> verts,
                                        MutableSpan<MEdge> edges,
                                        MutableSpan<MLoop> loops,
                                        MutableSpan<MPoly> polys,
                                        int &vert_offset,
                                        int &edge_offset,
                                        int &loop_offset,
                                        int &poly_offset)
{
  Span<float3> positions = spline.evaluated_positions();
  Span<float3> profile_positions = profile_spline.evaluated_positions();

  if (positions.size() == 0) {
    return;
  }

  if (profile_spline.size() == 1) {
    vert_extrude_to_mesh_data(
        spline, profile_positions[0], verts, edges, vert_offset, edge_offset);
    return;
  }

  /* TODO: This code path isn't finished, crashes, and needs more thought. */

  Array<float3> profile(profile_positions);

  const int vert_offset_start = vert_offset;

  for (const int i : IndexRange(positions.size() - 1)) {
    const float3 delta = positions[i + 1] - positions[i];
    for (float3 &profile_co : profile) {
      MVert &vert = verts[vert_offset++];
      copy_v3_v3(vert.co, profile_co);
      profile_co += delta;
    }
  }

  const int profile_len = profile.size();
  for (const int i : IndexRange(positions.size() - 1)) {
    const int ring_offset = vert_offset_start + profile_len * i;
    const int next_ring_offset = vert_offset_start + profile_len * (i + 1);
    for (const int UNUSED(i_profile) : profile.index_range()) {
      MEdge &edge_v = edges[edge_offset++];
      edge_v.v1 = ring_offset + i;
      edge_v.v2 = next_ring_offset + i;
      edge_v.flag = ME_LOOSEEDGE | ME_EDGEDRAW | ME_EDGERENDER;

      if (profile_len > 1) {
        MEdge &edge_u = edges[edge_offset++];
        edge_u.v1 = ring_offset + i;
        edge_u.v2 = ring_offset + (i + 1) % profile_len;
        edge_u.flag = ME_LOOSEEDGE | ME_EDGEDRAW | ME_EDGERENDER;
      }
    }
  }
}

static Mesh *curve_to_mesh_calculate(const DCurve &curve, const DCurve &profile_curve)
{
  int profile_vert_total = 0;
  int profile_edge_total = 0;
  for (const Spline *spline : profile_curve.splines) {
    Span<float3> positions = spline->evaluated_positions();
    profile_vert_total += positions.size();
    profile_edge_total += std::max(positions.size() - 2, 0L);
  }

  int vert_total = 0;
  int edge_total = 0;
  int poly_total = 0;
  for (const int i : curve.splines.index_range()) {
    const Spline &spline = *curve.splines[i];
    const int spline_vert_len = spline.evaluated_points_size();
    const int spline_edge_len = spline.is_cyclic ? spline_vert_len : (spline_vert_len - 1);
    /* An edge for every point for every curve segment, and edges for for the original profile's
     * edges. */
    vert_total += spline_vert_len * profile_vert_total;
    edge_total += spline_edge_len * profile_vert_total + spline_vert_len * profile_edge_total;
    poly_total += spline_edge_len * profile_edge_total;
  }
  const int corner_total = poly_total * 4;

  if (vert_total == 0) {
    return nullptr;
  }

  // Mesh *mesh = BKE_mesh_new_nomain(vert_total, edge_total, 0, corner_total, poly_total);
  Mesh *mesh = BKE_mesh_new_nomain(vert_total, edge_total, 0, 0, 0);
  MutableSpan<MVert> verts{mesh->mvert, mesh->totvert};
  MutableSpan<MEdge> edges{mesh->medge, mesh->totedge};
  MutableSpan<MLoop> loops{mesh->mloop, mesh->totloop};
  MutableSpan<MPoly> polys{mesh->mpoly, mesh->totpoly};

  int vert_offset = 0;
  int edge_offset = 0;
  int loop_offset = 0;
  int poly_offset = 0;
  for (const Spline *spline : curve.splines) {
    for (const Spline *profile_spline : profile_curve.splines) {
      spline_extrude_to_mesh_data(*spline,
                                  *profile_spline,
                                  verts,
                                  edges,
                                  loops,
                                  polys,
                                  vert_offset,
                                  edge_offset,
                                  loop_offset,
                                  poly_offset);
    }
  }

  BKE_mesh_calc_normals(mesh);
  BLI_assert(BKE_mesh_is_valid(mesh));

  return mesh;
}

static DCurve get_curve_single_vert()
{
  DCurve curve;
  BezierSpline *spline = new BezierSpline();
  BezierPoint control_point;
  control_point.position = float3(0);
  control_point.handle_position_a = float3(0);
  control_point.handle_position_b = float3(0);
  spline->control_points.append(control_point);
  curve.splines.append(static_cast<Spline *>(spline));

  return curve;
}

static void geo_node_curve_to_mesh_exec(GeoNodeExecParams params)
{
  GeometrySet curve_set = params.extract_input<GeometrySet>("Curve");
  GeometrySet profile_set = params.extract_input<GeometrySet>("Profile Curve");

  if (!curve_set.has_curve()) {
    params.set_output("Mesh", GeometrySet());
  }

  const DCurve *profile_curve = profile_set.get_curve_for_read();

  const DCurve vert_curve = get_curve_single_vert();

  Mesh *mesh = curve_to_mesh_calculate(*curve_set.get_curve_for_read(),
                                       (profile_curve == nullptr) ? vert_curve : *profile_curve);
  params.set_output("Mesh", GeometrySet::create_with_mesh(mesh));
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
