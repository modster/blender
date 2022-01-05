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

#include "BLI_task.hh"

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"

#include "BKE_mesh.h"
#include "BKE_spline.hh"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_input_normal_cc {

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_output<decl::Vector>(N_("Normal")).field_source();
}

static VArray<float3> construct_mesh_normals_gvarray(const MeshComponent &mesh_component,
                                                     const Mesh &mesh,
                                                     const IndexMask mask,
                                                     const AttributeDomain domain)
{
  switch (domain) {
    case ATTR_DOMAIN_FACE: {
      return VArray<float3>::ForSpan(
          {(float3 *)BKE_mesh_poly_normals_ensure(&mesh), mesh.totpoly});
    }
    case ATTR_DOMAIN_POINT: {
      return VArray<float3>::ForSpan(
          {(float3 *)BKE_mesh_vertex_normals_ensure(&mesh), mesh.totvert});
    }
    case ATTR_DOMAIN_EDGE: {
      /* In this case, start with vertex normals and convert to the edge domain, since the
       * conversion from edges to vertices is very simple. Use "manual" domain interpolation
       * instead of the GeometryComponent API to avoid calculating unnecessary values and to
       * allow normalizing the result more simply. */
      Span<float3> vert_normals{(float3 *)BKE_mesh_vertex_normals_ensure(&mesh), mesh.totvert};
      Array<float3> edge_normals(mask.min_array_size());
      Span<MEdge> edges{mesh.medge, mesh.totedge};
      for (const int i : mask) {
        const MEdge &edge = edges[i];
        edge_normals[i] =
            float3::interpolate(vert_normals[edge.v1], vert_normals[edge.v2], 0.5f).normalized();
      }

      return VArray<float3>::ForContainer(std::move(edge_normals));
    }
    case ATTR_DOMAIN_CORNER: {
      /* The normals on corners are just the mesh's face normals, so start with the face normal
       * array and copy the face normal for each of its corners. In this case using the mesh
       * component's generic domain interpolation is fine, the data will still be normalized,
       * since the face normal is just copied to every corner. */
      return mesh_component.attribute_try_adapt_domain(
          VArray<float3>::ForSpan({(float3 *)BKE_mesh_poly_normals_ensure(&mesh), mesh.totpoly}),
          ATTR_DOMAIN_FACE,
          ATTR_DOMAIN_CORNER);
    }
    default:
      return {};
  }
}

static void calculate_bezier_normals(const BezierSpline &spline, MutableSpan<float3> normals)
{
  Span<int> offsets = spline.control_point_offsets();
  Span<float3> evaluated_normals = spline.evaluated_normals();
  for (const int i : IndexRange(spline.size())) {
    normals[i] = evaluated_normals[offsets[i]];
  }
}

static void calculate_poly_normals(const PolySpline &spline, MutableSpan<float3> normals)
{
  normals.copy_from(spline.evaluated_normals());
}

/**
 * Because NURBS control points are not necessarily on the path, the normal at the control points
 * is not well defined, so create a temporary poly spline to find the normals. This requires extra
 * copying currently, but may be more efficient in the future if attributes have some form of CoW.
 */
static void calculate_nurbs_normals(const NURBSpline &spline, MutableSpan<float3> normals)
{
  PolySpline poly_spline;
  poly_spline.resize(spline.size());
  poly_spline.positions().copy_from(spline.positions());
  normals.copy_from(poly_spline.evaluated_normals());
}

static Array<float3> curve_normal_point_domain(const CurveEval &curve)
{
  Span<SplinePtr> splines = curve.splines();
  Array<int> offsets = curve.control_point_offsets();
  const int total_size = offsets.last();
  Array<float3> normals(total_size);

  threading::parallel_for(splines.index_range(), 128, [&](IndexRange range) {
    for (const int i : range) {
      const Spline &spline = *splines[i];
      MutableSpan spline_normals{normals.as_mutable_span().slice(offsets[i], spline.size())};
      switch (splines[i]->type()) {
        case Spline::Type::Bezier:
          calculate_bezier_normals(static_cast<const BezierSpline &>(spline), spline_normals);
          break;
        case Spline::Type::Poly:
          calculate_poly_normals(static_cast<const PolySpline &>(spline), spline_normals);
          break;
        case Spline::Type::NURBS:
          calculate_nurbs_normals(static_cast<const NURBSpline &>(spline), spline_normals);
          break;
      }
    }
  });
  return normals;
}

static VArray<float3> construct_curve_normal_gvarray(const CurveComponent &component,
                                                     const AttributeDomain domain)
{
  const CurveEval *curve = component.get_for_read();
  if (curve == nullptr) {
    return nullptr;
  }

  if (domain == ATTR_DOMAIN_POINT) {
    const Span<SplinePtr> splines = curve->splines();

    /* Use a reference to evaluated normals if possible to avoid an allocation and a copy.
     * This is only possible when there is only one poly spline. */
    if (splines.size() == 1 && splines.first()->type() == Spline::Type::Poly) {
      const PolySpline &spline = static_cast<PolySpline &>(*splines.first());
      return VArray<float3>::ForSpan(spline.evaluated_normals());
    }

    Array<float3> normals = curve_normal_point_domain(*curve);
    return VArray<float3>::ForContainer(std::move(normals));
  }

  if (domain == ATTR_DOMAIN_CURVE) {
    Array<float3> point_normals = curve_normal_point_domain(*curve);
    VArray<float3> varray = VArray<float3>::ForContainer(std::move(point_normals));
    return component.attribute_try_adapt_domain<float3>(
        std::move(varray), ATTR_DOMAIN_POINT, ATTR_DOMAIN_CURVE);
  }

  return nullptr;
}

class NormalFieldInput final : public GeometryFieldInput {
 public:
  NormalFieldInput() : GeometryFieldInput(CPPType::get<float3>(), "Normal node")
  {
    category_ = Category::Generated;
  }

  GVArray get_varray_for_context(const GeometryComponent &component,
                                 const AttributeDomain domain,
                                 IndexMask mask) const final
  {
    if (component.type() == GEO_COMPONENT_TYPE_MESH) {
      const MeshComponent &mesh_component = static_cast<const MeshComponent &>(component);
      const Mesh *mesh = mesh_component.get_for_read();
      if (mesh == nullptr) {
        return {};
      }

      return construct_mesh_normals_gvarray(mesh_component, *mesh, mask, domain);
    }
    if (component.type() == GEO_COMPONENT_TYPE_CURVE) {
      const CurveComponent &curve_component = static_cast<const CurveComponent &>(component);
      return construct_curve_normal_gvarray(curve_component, domain);
    }
    return {};
  }

  uint64_t hash() const override
  {
    /* Some random constant hash. */
    return 669605641;
  }

  bool is_equal_to(const fn::FieldNode &other) const override
  {
    return dynamic_cast<const NormalFieldInput *>(&other) != nullptr;
  }
};

static void node_geo_exec(GeoNodeExecParams params)
{
  Field<float3> normal_field{std::make_shared<NormalFieldInput>()};
  params.set_output("Normal", std::move(normal_field));
}

}  // namespace blender::nodes::node_geo_input_normal_cc

void register_node_type_geo_input_normal()
{
  namespace file_ns = blender::nodes::node_geo_input_normal_cc;

  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_INPUT_NORMAL, "Normal", NODE_CLASS_INPUT);
  ntype.geometry_node_execute = file_ns::node_geo_exec;
  ntype.declare = file_ns::node_declare;
  nodeRegisterType(&ntype);
}
