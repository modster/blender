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

#include "BLI_array.hh"
#include "BLI_float4x4.hh"
#include "BLI_resource_scope.hh"
#include "BLI_task.hh"
#include "BLI_timeit.hh"

#include "BKE_attribute_math.hh"
#include "BKE_geometry_set_instances.hh"
#include "BKE_spline.hh"

#include "UI_interface.h"
#include "UI_resources.h"

#include "node_geometry_util.hh"

using blender::float4x4;
using blender::bke::GeometryInstanceGroup;

static bNodeSocketTemplate geo_node_curve_deform_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_GEOMETRY, N_("Curve")},
    {SOCK_STRING, N_("Factor")},
    {SOCK_FLOAT, N_("Factor"), 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, PROP_FACTOR},
    {SOCK_BOOLEAN, N_("Use Bounds")},
    {SOCK_BOOLEAN, N_("Stretch")},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_curve_deform_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

static void geo_node_curve_deform_layout(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
{
  uiItemR(layout, ptr, "axis", UI_ITEM_R_EXPAND, nullptr, ICON_NONE);
}

static void geo_node_curve_deform_init(bNodeTree *UNUSED(tree), bNode *node)
{
  NodeGeometryCurveDeform *data = (NodeGeometryCurveDeform *)MEM_callocN(
      sizeof(NodeGeometryCurveDeform), __func__);

  data->axis = GEO_NODE_CURVE_DEFORM_POSX;
  node->storage = data;
}

namespace blender::nodes {

static constexpr int deform_axis_index(const GeometryNodeCurveDeformAxis axis)
{
  switch (axis) {
    case GEO_NODE_CURVE_DEFORM_POSX:
    case GEO_NODE_CURVE_DEFORM_NEGX:
      return 0;
    case GEO_NODE_CURVE_DEFORM_POSY:
    case GEO_NODE_CURVE_DEFORM_NEGY:
      return 1;
    case GEO_NODE_CURVE_DEFORM_POSZ:
    case GEO_NODE_CURVE_DEFORM_NEGZ:
      return 2;
  }
  BLI_assert_unreachable();
  return 0;
}

static constexpr int deform_next_axis_index(const GeometryNodeCurveDeformAxis axis)
{
  switch (axis) {
    case GEO_NODE_CURVE_DEFORM_POSX:
    case GEO_NODE_CURVE_DEFORM_NEGX:
      return 1;
    case GEO_NODE_CURVE_DEFORM_POSY:
    case GEO_NODE_CURVE_DEFORM_NEGY:
      return 2;
    case GEO_NODE_CURVE_DEFORM_POSZ:
    case GEO_NODE_CURVE_DEFORM_NEGZ:
      return 0;
  }
  BLI_assert_unreachable();
  return 0;
}

static constexpr int deform_other_axis_index(const GeometryNodeCurveDeformAxis axis)
{
  switch (axis) {
    case GEO_NODE_CURVE_DEFORM_POSX:
    case GEO_NODE_CURVE_DEFORM_NEGX:
      return 2;
    case GEO_NODE_CURVE_DEFORM_POSY:
    case GEO_NODE_CURVE_DEFORM_NEGY:
      return 0;
    case GEO_NODE_CURVE_DEFORM_POSZ:
    case GEO_NODE_CURVE_DEFORM_NEGZ:
      return 1;
  }
  BLI_assert_unreachable();
  return 0;
}

static constexpr bool axis_is_negative(const GeometryNodeCurveDeformAxis axis)
{
  switch (axis) {
    case GEO_NODE_CURVE_DEFORM_POSX:
    case GEO_NODE_CURVE_DEFORM_POSY:
    case GEO_NODE_CURVE_DEFORM_POSZ:
      return false;
    case GEO_NODE_CURVE_DEFORM_NEGX:
    case GEO_NODE_CURVE_DEFORM_NEGY:
    case GEO_NODE_CURVE_DEFORM_NEGZ:
      return true;
  }
  BLI_assert_unreachable();
  return false;
}

struct SplineDeformInput {
  const Spline &spline;
  Span<float3> positions;
  Span<float3> tangents;
  Span<float3> normals;
  GVArray_Typed<float> radii;
  float total_length;
  bool use_stretch;
  bool use_bounds;
};

static float3 deform_position(const SplineDeformInput &in,
                              const Spline::LookupResult &lookup,
                              const float cotangent_factor,
                              const float normal_factor,
                              const bool is_negative)
{
  const int index = lookup.evaluated_index;
  const int next = lookup.next_evaluated_index;
  const float factor = lookup.factor;
  const float clamped = std::clamp(lookup.factor, 0.0f, 1.0f);

  const float3 position = float3::interpolate(in.positions[index], in.positions[next], factor);
  const float3 tangent = float3::interpolate(in.tangents[index], in.tangents[next], clamped);
  const float3 normal = float3::interpolate(in.normals[index], in.normals[next], clamped);
  const float3 cotangent = float3::cross(tangent, normal).normalized();
  const float radius = interpf(in.radii[next], in.radii[index], clamped);

  if (is_negative) {
    return position + (cotangent * cotangent_factor + normal * normal_factor) * radius;
  }

  return position - (cotangent * cotangent_factor + normal * normal_factor) * radius;
}

struct Bounds {
  float3 min;
  float3 max;
  float3 inv_size;
};

static Bounds position_bounds(const Span<float3> positions)
{
  float3 min = float3(FLT_MAX);
  float3 max = float3(-FLT_MAX);
  for (const float3 &position : positions) {
    minmax_v3v3_v3(min, max, position);
  }
  return {min, max, float3::safe_divide(float3(1), max - min)};
}

static Bounds dummy_parameter_bounds(const GeometryNodeCurveDeformAxis deform_axis)
{
  if (axis_is_negative(deform_axis)) {
    return {float3(-1), float3(0), float3(-1)};
  }
  return {float3(0), float3(1), float3(1)};
}

static float process_parameter(const float3 position,
                               const int axis_index,
                               const bool is_negative,
                               const SplineDeformInput &input,
                               const Bounds &bounds)
{
  const float parameter = is_negative ? -(position[axis_index] - bounds.max[axis_index]) :
                                        position[axis_index] - bounds.min[axis_index];
  if (input.use_stretch) {
    return parameter * bounds.inv_size[axis_index] * input.total_length;
  }
  return parameter;
}

static void execute_on_component(const GeoNodeExecParams &params,
                                 const SplineDeformInput &input,
                                 GeometryComponent &component)
{
  const NodeGeometryCurveDeform &node_storage = *(NodeGeometryCurveDeform *)params.node().storage;
  const GeometryNodeCurveDeformAxis deform_axis = (GeometryNodeCurveDeformAxis)node_storage.axis;
  const int axis_index = deform_axis_index(deform_axis);
  const int next_axis = deform_next_axis_index(deform_axis);
  const int other_axis = deform_other_axis_index(deform_axis);
  const bool is_negative = axis_is_negative(deform_axis);

  OutputAttribute_Typed<float3> position_attribute =
      component.attribute_try_get_for_output<float3>("position", ATTR_DOMAIN_POINT, {0, 0, 0});
  MutableSpan<float3> positions = position_attribute.as_span();
  const Bounds bounds = position_bounds(positions);
  const Bounds parameter_bounds = input.use_bounds ? bounds : dummy_parameter_bounds(deform_axis);

  parallel_for(positions.index_range(), 1024, [&](IndexRange range) {
    for (const int i : range) {
      const float parameter = process_parameter(
          positions[i], axis_index, is_negative, input, parameter_bounds);
      std::cout << "Parameter: " << parameter << "\n";
      const Spline::LookupResult lookup = input.spline.lookup_evaluated_length(parameter);

      const float3 co = (positions[i] - bounds.min) * bounds.inv_size * 2.0f - float3(1);
      if (is_negative) {
        positions[i] = deform_position(input, lookup, co[next_axis], co[other_axis], is_negative);
      }
      else {
        positions[i] = deform_position(input, lookup, co[other_axis], co[next_axis], is_negative);
      }
    }
  });

  position_attribute.save();
}

static void geo_node_curve_deform_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");
  GeometrySet curve_geometry_set = params.extract_input<GeometrySet>("Curve");

  geometry_set = bke::geometry_set_realize_instances(geometry_set);

  /* TODO: Theoretically this could be easily avoided. */
  curve_geometry_set = bke::geometry_set_realize_instances(curve_geometry_set);

  const CurveEval *curve = curve_geometry_set.get_curve_for_read();
  if (curve == nullptr || curve->splines().size() == 0) {
    params.set_output("Geometry", geometry_set);
    return;
  }

  const Spline &spline = *curve->splines().first();
  const float total_length = spline.length();
  if (total_length == 0.0f) {
    params.set_output("Geometry", geometry_set);
    return;
  }

  const SplineDeformInput spline_data{spline,
                                      spline.evaluated_positions(),
                                      spline.evaluated_tangents(),
                                      spline.evaluated_normals(),
                                      spline.interpolate_to_evaluated_points(spline.radii()),
                                      total_length,
                                      params.extract_input<bool>("Stretch"),
                                      params.extract_input<bool>("Use Bounds")};

  if (geometry_set.has<MeshComponent>()) {
    execute_on_component(
        params, spline_data, geometry_set.get_component_for_write<MeshComponent>());
  }
  if (geometry_set.has<PointCloudComponent>()) {
    execute_on_component(
        params, spline_data, geometry_set.get_component_for_write<PointCloudComponent>());
  }
  if (geometry_set.has<CurveComponent>()) {
    execute_on_component(
        params, spline_data, geometry_set.get_component_for_write<CurveComponent>());
  }

  params.set_output("Geometry", geometry_set);
}

}  // namespace blender::nodes

void register_node_type_geo_curve_deform()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_CURVE_DEFORM, "Curve Deform", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, geo_node_curve_deform_in, geo_node_curve_deform_out);
  ntype.draw_buttons = geo_node_curve_deform_layout;
  node_type_storage(
      &ntype, "NodeGeometryCurveDeform", node_free_standard_storage, node_copy_standard_storage);
  node_type_init(&ntype, geo_node_curve_deform_init);
  ntype.geometry_node_execute = blender::nodes::geo_node_curve_deform_exec;
  nodeRegisterType(&ntype);
}
