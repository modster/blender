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
    {SOCK_BOOLEAN, N_("Stretch to Fit")},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_curve_deform_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

static void geo_node_curve_deform_layout(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
{
  const bNode *node = (bNode *)ptr->data;
  NodeGeometryCurveDeform &node_storage = *(NodeGeometryCurveDeform *)node->storage;
  const GeometryNodeCurveDeformMode mode = (GeometryNodeCurveDeformMode)node_storage.input_mode;

  uiItemR(layout, ptr, "axis", UI_ITEM_R_EXPAND, nullptr, ICON_NONE);

  uiItemR(layout, ptr, "input_mode", UI_ITEM_R_EXPAND, nullptr, ICON_NONE);

  if (mode == GEO_NODE_CURVE_DEFORM_ATTRIBUTE) {
    uiLayoutSetPropSep(layout, true);
    uiLayoutSetPropDecorate(layout, false);
    uiItemR(layout, ptr, "attribute_input_type", 0, IFACE_("Factor"), ICON_NONE);
  }
}

static void geo_node_curve_deform_init(bNodeTree *UNUSED(tree), bNode *node)
{
  NodeGeometryCurveDeform *data = (NodeGeometryCurveDeform *)MEM_callocN(
      sizeof(NodeGeometryCurveDeform), __func__);

  data->input_mode = GEO_NODE_CURVE_DEFORM_POSITION;
  data->axis = GEO_NODE_CURVE_DEFORM_POSX;
  data->attribute_input_type = GEO_NODE_ATTRIBUTE_INPUT_ATTRIBUTE;
  node->storage = data;
}

namespace blender::nodes {

static void geo_node_curve_deform_update(bNodeTree *UNUSED(ntree), bNode *node)
{
  NodeGeometryCurveDeform &node_storage = *(NodeGeometryCurveDeform *)node->storage;
  const GeometryNodeCurveDeformMode mode = (GeometryNodeCurveDeformMode)node_storage.input_mode;

  bNodeSocket *attribute_socket = ((bNodeSocket *)node->inputs.first)->next->next;

  nodeSetSocketAvailability(attribute_socket, mode == GEO_NODE_CURVE_DEFORM_ATTRIBUTE);
  update_attribute_input_socket_availabilities(
      *node,
      "Factor",
      (GeometryNodeAttributeInputMode)node_storage.attribute_input_type,
      mode == GEO_NODE_CURVE_DEFORM_ATTRIBUTE);
}

enum class Axis { X, Y, Z };
static Axis axis_from_deform_axis(const GeometryNodeCurveDeformAxis axis)
{
  switch (axis) {
    case GEO_NODE_CURVE_DEFORM_POSX:
    case GEO_NODE_CURVE_DEFORM_NEGX:
      return Axis::X;
    case GEO_NODE_CURVE_DEFORM_POSY:
    case GEO_NODE_CURVE_DEFORM_NEGY:
      return Axis::Y;
    case GEO_NODE_CURVE_DEFORM_POSZ:
    case GEO_NODE_CURVE_DEFORM_NEGZ:
      return Axis::Z;
  }
  BLI_assert_unreachable();
  return Axis::X;
}

static bool axis_is_negative(const GeometryNodeCurveDeformAxis axis)
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

struct SplineDataInput {
  const Spline &spline;
  Span<float3> positions;
  Span<float3> tangents;
  Span<float3> normals;
  GVArray_Typed<float> radii;
};

static float3 deform_position(const SplineDataInput &in,
                              const float index_factor,
                              const float cotangent_factor,
                              const float normal_factor)
{
  const Spline::LookupResult interp = in.spline.lookup_data_from_index_factor(index_factor);
  const int index = interp.evaluated_index;
  const int next = interp.next_evaluated_index;
  const float factor = interp.factor;

  const float3 position = float3::interpolate(in.positions[index], in.positions[next], factor);
  const float3 tangent = float3::interpolate(in.tangents[index], in.tangents[next], factor);
  const float3 normal = float3::interpolate(in.normals[index], in.normals[next], factor);
  const float3 cotangent = float3::cross(tangent, normal).normalized();
  const float radius = interpf(in.radii[next], in.radii[index], factor);

  return position + (cotangent * cotangent_factor + normal * normal_factor) * radius;
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

static void spline_deform(const SplineDataInput &spline_data,
                          const Span<float> index_factors,
                          const Span<int> indices,
                          const Axis axis,
                          const Bounds &bounds,
                          MutableSpan<float3> positions)
{
  switch (axis) {
    case Axis::X:
      parallel_for(positions.index_range(), 1024, [&](IndexRange range) {
        for (const int i : range) {
          const float3 co = (positions[indices[i]] - bounds.min) * bounds.inv_size - float3(0.5f);
          positions[indices[i]] = deform_position(spline_data, index_factors[i], co.y, co.z);
        }
      });
      break;
    case Axis::Y:
      parallel_for(positions.index_range(), 1024, [&](IndexRange range) {
        for (const int i : range) {
          const float3 co = (positions[indices[i]] - bounds.min) * bounds.inv_size - float3(0.5f);
          positions[indices[i]] = deform_position(spline_data, index_factors[i], co.z, co.x);
        }
      });
      break;
    case Axis::Z:
      parallel_for(positions.index_range(), 1024, [&](IndexRange range) {
        for (const int i : range) {
          const float3 co = (positions[indices[i]] - bounds.min) * bounds.inv_size - float3(0.5f);
          positions[indices[i]] = deform_position(spline_data, index_factors[i], co.x, co.y);
        }
      });
      break;
  }
}

static void retrieve_position_parameters(const Span<float3> positions,
                                         const Axis axis,
                                         MutableSpan<float> parameters,
                                         MutableSpan<int> indices)
{
  Span co{positions};
  switch (axis) {
    case Axis::X:
      std::sort(indices.begin(), indices.end(), [&](int a, int b) { return co[a].x < co[b].x; });
      parallel_for(IndexRange(positions.size()), 2048, [&](IndexRange range) {
        for (const int i : range) {
          parameters[i] = positions[indices[i]].x;
        }
      });
      break;
    case Axis::Y:
      std::sort(indices.begin(), indices.end(), [&](int a, int b) { return co[a].y < co[b].y; });
      parallel_for(IndexRange(positions.size()), 2048, [&](IndexRange range) {
        for (const int i : range) {
          parameters[i] = positions[indices[i]].y;
        }
      });
      break;
    case Axis::Z:
      std::sort(indices.begin(), indices.end(), [&](int a, int b) { return co[a].z < co[b].z; });
      parallel_for(IndexRange(positions.size()), 2048, [&](IndexRange range) {
        for (const int i : range) {
          parameters[i] = positions[indices[i]].z;
        }
      });
      break;
  }
}

static void retrieve_attribute_parameters(const GVArray_Typed<float> attribute,
                                          const float total_length,
                                          MutableSpan<float> parameters,
                                          MutableSpan<int> indices)
{
  VArray_Span<float> span{*attribute};

  std::sort(indices.begin(), indices.end(), [&](int a, int b) { return span[a] < span[b]; });

  parallel_for(IndexRange(attribute.size()), 2048, [&](IndexRange range) {
    for (const int i : range) {
      parameters[i] = span[indices[i]] * total_length;
    }
  });
}

static void process_parameters(const GeoNodeExecParams &params,
                               const GeometryNodeCurveDeformAxis deform_axis,
                               const float total_length,
                               MutableSpan<float> parameters,
                               MutableSpan<int> indices)
{
  const int size = parameters.size();
  if (params.get_input<bool>("Stretch to Fit")) {
    const double min = parameters.first();
    const double max = parameters.last();
    const double parameter_range = max - min;
    const double factor = (parameter_range == 0.0f) ? 0.0f : total_length / parameter_range;
    parallel_for(IndexRange(size), 2048, [&](IndexRange range) {
      for (const int i : range) {
        parameters[i] = (double(parameters[i]) - min) * factor;
      }
    });
    /* Prevent overflow in some cases. */
    parameters.last() = total_length;
  }
  else {
    parallel_for(IndexRange(size), 2048, [&](IndexRange range) {
      for (const int i : range) {
        parameters[i] = std::clamp(parameters[i], 0.0f, total_length);
      }
    });
  }

  /* TODO: Broken. */
  /* Reverse parameters if necessary (also the indices to maintain the sorted input to deform). */
  if (axis_is_negative(deform_axis)) {
    parallel_for(IndexRange(size), 1024, [&](IndexRange range) {
      for (const int i : range) {
        parameters[i] = total_length - parameters[size - i - 1];
        indices[i] = indices[size - i - 1];
      }
    });
  }
}

// struct ExtrapolationSpans {
//   Span<int> start;
//   Span<int> spline;
//   Span<int> end;
// };
// static ExtrapolationSpans find_extrapolation_points(const Span<float> parameters,
//                                                     const Span<int> indices,
//                                                     const float total_length)
// {
//   const float *start = std::lower_bound(parameters.begin(), parameters.end(), 0.0f);
//   const float *end = std::lower_bound(parameters.begin(), parameters.end(), total_length);
//   const int start_index = start - parameters.begin();
//   const int end_index = end - parameters.begin();

//   return {indices.take_front(start_index),
//           indices.slice(start_index, end_index - start_index),
//           indices.take_back(indices.size() - end_index)};
// }

static void execute_on_component(const GeoNodeExecParams &params,
                                 const CurveEval &curve,
                                 GeometryComponent &component)
{
  const NodeGeometryCurveDeform &node_storage = *(NodeGeometryCurveDeform *)params.node().storage;
  const GeometryNodeCurveDeformMode mode = (GeometryNodeCurveDeformMode)node_storage.input_mode;
  const GeometryNodeCurveDeformAxis deform_axis = (GeometryNodeCurveDeformAxis)node_storage.axis;
  const Axis axis = axis_from_deform_axis(deform_axis);

  const Spline &spline = *curve.splines().first();
  const float total_length = spline.length();

  const int size = component.attribute_domain_size(ATTR_DOMAIN_POINT);
  OutputAttribute_Typed<float3> position_attribute =
      component.attribute_try_get_for_output<float3>("position", ATTR_DOMAIN_POINT, {0, 0, 0});
  MutableSpan<float3> positions = position_attribute.as_span();
  const Bounds bounds = position_bounds(positions);

  /* #sample_length_parameters_to_index_factors requires an array of sorted parameters.
   * Sort indices based on the parameters before processing, build the parameters final
   * parameters, then use the indices to map back to the orignal positions. */
  Array<float> parameters(size);
  Array<int> sorted_indices(size);
  for (const int i : sorted_indices.index_range()) {
    sorted_indices[i] = i;
  }

  switch (mode) {
    case GEO_NODE_CURVE_DEFORM_POSITION: {
      retrieve_position_parameters(positions, axis, parameters, sorted_indices);
      break;
    }
    case GEO_NODE_CURVE_DEFORM_ATTRIBUTE: {
      retrieve_attribute_parameters(
          params.get_input_attribute<float>("Factor", component, ATTR_DOMAIN_POINT, 0.0f),
          total_length,
          parameters,
          sorted_indices);
      break;
    }
  }

  process_parameters(params, deform_axis, total_length, parameters, sorted_indices);

  const SplineDataInput spline_data{spline,
                                    spline.evaluated_positions(),
                                    spline.evaluated_tangents(),
                                    spline.evaluated_normals(),
                                    spline.interpolate_to_evaluated_points(spline.radii())};

  // const ExtrapolationSpans index_spans = find_extrapolation_points(
  //     parameters, sorted_indices, total_length);

  spline.sample_length_parameters_to_index_factors(parameters);
  spline_deform(spline_data, parameters, sorted_indices, axis, bounds, positions);
  // deform_start_extrapolation(spline_data, parameters, index_spans.start, axis);

  position_attribute.save();
}

static void geo_node_curve_deform_exec(GeoNodeExecParams params)
{
  GeometrySet deform_geometry_set = params.extract_input<GeometrySet>("Geometry");
  GeometrySet curve_geometry_set = params.extract_input<GeometrySet>("Curve");

  deform_geometry_set = bke::geometry_set_realize_instances(deform_geometry_set);

  /* TODO: Theoretically this could be easily avoided. */
  curve_geometry_set = bke::geometry_set_realize_instances(curve_geometry_set);

  const CurveEval *curve = curve_geometry_set.get_curve_for_read();
  if (curve == nullptr || curve->splines().size() == 0) {
    params.set_output("Geometry", deform_geometry_set);
    return;
  }

  if (deform_geometry_set.has<MeshComponent>()) {
    execute_on_component(
        params, *curve, deform_geometry_set.get_component_for_write<MeshComponent>());
  }
  if (deform_geometry_set.has<PointCloudComponent>()) {
    execute_on_component(
        params, *curve, deform_geometry_set.get_component_for_write<PointCloudComponent>());
  }
  if (deform_geometry_set.has<CurveComponent>()) {
    execute_on_component(
        params, *curve, deform_geometry_set.get_component_for_write<CurveComponent>());
  }

  params.set_output("Geometry", deform_geometry_set);
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
  node_type_update(&ntype, blender::nodes::geo_node_curve_deform_update);
  ntype.geometry_node_execute = blender::nodes::geo_node_curve_deform_exec;
  nodeRegisterType(&ntype);
}
