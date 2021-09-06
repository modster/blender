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

#include "UI_interface.h"
#include "UI_resources.h"

#include "DNA_node_types.h"

#include "node_geometry_util.hh"

#include "BKE_spline.hh"

static bNodeSocketTemplate geo_node_curve_fillet_in[] = {
    {SOCK_GEOMETRY, N_("Curve")},
    {SOCK_INT, N_("Poly Count"), 1, 0, 0, 0, 1, 1000},
    {SOCK_BOOLEAN, N_("Limit Radius")},
    {SOCK_FLOAT, N_("Radius"), 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, FLT_MAX, PROP_DISTANCE},
    {SOCK_STRING, N_("Radius")},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_curve_fillet_out[] = {
    {SOCK_GEOMETRY, N_("Curve")},
    {-1, ""},
};

static void geo_node_curve_fillet_layout(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
{
  uiLayoutSetPropSep(layout, true);
  uiLayoutSetPropDecorate(layout, false);
  uiItemR(layout, ptr, "mode", 0, IFACE_("Mode"), ICON_NONE);
  uiItemR(layout, ptr, "radius_mode", 0, IFACE_("Radius Mode"), ICON_NONE);
}

static void geo_node_curve_fillet_init(bNodeTree *UNUSED(tree), bNode *node)
{
  NodeGeometryCurveFillet *data = (NodeGeometryCurveFillet *)MEM_callocN(
      sizeof(NodeGeometryCurveFillet), __func__);

  data->mode = GEO_NODE_CURVE_FILLET_BEZIER;
  data->radius_mode = GEO_NODE_ATTRIBUTE_INPUT_FLOAT;

  node->storage = data;
}

namespace blender::nodes {

struct FilletParam {
  GeometryNodeCurveFilletMode mode;

  /* Number of points to be added. */
  std::optional<int> count;

  /* Whether or not fillets are allowed to overlap. */
  bool limit_radius;

  /* Radii for fillet arc at all vertices. */
  GVArray_Typed<float> *radii;
};

/* A data structure used to store fillet data about all vertices to be filleted. */
struct FilletData {
  Array<float3> directions, positions, axes;
  Array<float> radii, angles;
  Array<int> counts;

  FilletData(const int size)
  {
    directions.reinitialize(size);
    positions.reinitialize(size);
    axes.reinitialize(size);
    radii.reinitialize(size);
    angles.reinitialize(size);
    counts.reinitialize(size);
  }
};

static void geo_node_curve_fillet_update(bNodeTree *UNUSED(ntree), bNode *node)
{
  NodeGeometryCurveFillet &node_storage = *(NodeGeometryCurveFillet *)node->storage;
  const GeometryNodeCurveFilletMode mode = (GeometryNodeCurveFilletMode)node_storage.mode;

  bNodeSocket *poly_socket = ((bNodeSocket *)node->inputs.first)->next;

  nodeSetSocketAvailability(poly_socket, mode == GEO_NODE_CURVE_FILLET_POLY);

  update_attribute_input_socket_availabilities(
      *node, "Radius", (GeometryNodeAttributeInputMode)node_storage.radius_mode);
}

/* Function to get the center of a fillet. */
static float3 get_center(const float3 vec_pos2prev,
                         const float3 pos,
                         const float3 axis,
                         const float angle)
{
  float3 vec_pos2center;
  rotate_v3_v3v3fl(vec_pos2center, vec_pos2prev, axis, M_PI_2 - angle / 2.0f);
  vec_pos2center *= 1.0f / sinf(angle / 2.0f);

  return vec_pos2center + pos;
}

/* Function to get the center of the fillet using fillet data */
static float3 get_center(const float3 vec_pos2prev, const FilletData &fd, const int index)
{
  const float angle = fd.angles[index];
  const float3 axis = fd.axes[index];
  const float3 pos = fd.positions[index];

  return get_center(vec_pos2prev, pos, axis, angle);
}

/* Calculate the direction vectors from each vertex to their previous vertex. */
static Array<float3> calculate_directions(const Span<float3> positions)
{
  const int size = positions.size();
  Array<float3> directions(size);

  for (const int i : IndexRange(size - 1)) {
    directions[i] = (positions[i + 1] - positions[i]).normalized();
  }
  directions[size - 1] = (positions[0] - positions[size - 1]).normalized();

  return directions;
}

/* Calculate the axes around which the fillet is built. */
static Array<float3> calculate_axes(const Span<float3> directions)
{
  const int size = directions.size();
  Array<float3> axes(size);

  axes[0] = float3::cross(-directions[size - 1], directions[0]);
  for (const int i : IndexRange(1, size - 1)) {
    axes[i] = float3::cross(-directions[i - 1], directions[i]);
  }

  return axes;
}

/* Calculate the angle of the arc formed by the fillet. */
static Array<float> calculate_angles(const Span<float3> directions)
{
  const int size = directions.size();
  Array<float> angles(size);

  angles[0] = M_PI - angle_v3v3(-directions[size - 1], directions[0]);
  for (const int i : IndexRange(1, size - 1)) {
    angles[i] = M_PI - angle_v3v3(-directions[i - 1], directions[i]);
  }

  return angles;
}

/* Calculate the segment count in each filleted arc. */
static Array<int> calculate_counts(const std::optional<int> count,
                                   const int size,
                                   const bool cyclic)
{
  Array<int> counts(size, *count);
  if (!cyclic) {
    counts[0] = counts[size - 1] = 0;
  }

  return counts;
}

/* Calculate the radii for the vertices to be filleted. */
static Array<float> calculate_radii(const FilletParam &fillet_param,
                                    const int size,
                                    const int spline_index)
{
  Array<float> radii(size, 0.0f);

  for (const int i : IndexRange(size)) {
    const float radius = (*fillet_param.radii)[spline_index + i];
    radii[i] = fillet_param.limit_radius && radius < 0.0f ? 0.0f : radius;
  }

  return radii;
}

/* Calculate the number of vertices added per vertex on the source spline. */
static int calculate_point_counts(MutableSpan<int> point_counts,
                                  const Span<float> radii,
                                  const Span<int> counts)
{
  int added_count = 0;
  for (const int i : IndexRange(point_counts.size())) {
    /* Calculate number of points to be added for the vertex. */
    if (radii[i] != 0.0f) {
      added_count += counts[i];
      point_counts[i] = counts[i] + 1;
    }
  }

  return added_count;
}

/* Function to calculate and obtain the fillet data for the entire spline. */
static FilletData calculate_fillet_data(const Spline &spline,
                                        const FilletParam &fillet_param,
                                        int &added_count,
                                        MutableSpan<int> point_counts,
                                        const int spline_index)
{
  const int size = spline.size();

  FilletData fd(size);
  fd.directions = calculate_directions(spline.positions());
  fd.positions = spline.positions();
  fd.axes = calculate_axes(fd.directions);
  fd.angles = calculate_angles(fd.directions);
  fd.counts = calculate_counts(fillet_param.count, size, spline.is_cyclic());
  fd.radii = calculate_radii(fillet_param, size, spline_index);

  added_count = calculate_point_counts(point_counts, fd.radii, fd.counts);

  return fd;
}

/* Limit the radius based on angle and radii to prevent overlap. */
static void limit_radii(FilletData &fd, const bool cyclic)
{
  MutableSpan<float> radii(fd.radii);
  Span<float> angles(fd.angles);
  Span<float3> positions(fd.positions);

  const int size = radii.size();
  int fillet_count, start;
  Array<float> max_radii(size, FLT_MAX);

  if (cyclic) {
    fillet_count = size;
    start = 0;

    /* Calculate lengths between adjacent control points. */
    const float len_prev = float3::distance(positions[0], positions[size - 1]);
    const float len_next = float3::distance(positions[0], positions[1]);

    /* Calculate tangent lengths of fillets in control points. */
    const float tan_len = radii[0] * tanf(angles[0] / 2.0f);
    const float tan_len_prev = radii[size - 1] * tanf(angles[size - 1] / 2.0f);
    const float tan_len_next = radii[1] * tanf(angles[1] / 2.0f);

    float factor_prev = 1.0f, factor_next = 1.0f;
    if (tan_len + tan_len_prev > len_prev) {
      factor_prev = len_prev / (tan_len + tan_len_prev);
    }
    if (tan_len + tan_len_next > len_next) {
      factor_next = len_next / (tan_len + tan_len_next);
    }

    /* Scale max radii by calculated factors. */
    max_radii[0] = radii[0] * min_ff(factor_next, factor_prev);
    max_radii[1] = radii[1] * factor_next;
    max_radii[size - 1] = radii[size - 1] * factor_prev;
  }
  else {
    fillet_count = size - 2;
    start = 1;
  }

  /* Initialize max_radii to largest possible radii. */
  float prev_dist = float3::distance(positions[1], positions[0]);
  for (const int i : IndexRange(1, size - 2)) {
    const float temp_dist = float3::distance(positions[i], positions[i + 1]);
    max_radii[i] = min_ff(prev_dist, temp_dist) / tanf(angles[i] / 2.0f);
    prev_dist = temp_dist;
  }

  /* Max radii calculations for each index. */
  for (const int i : IndexRange(start, fillet_count - 1)) {
    const float len_next = float3::distance(positions[i], positions[i + 1]);
    const float tan_len = radii[i] * tanf(angles[i] / 2.0f);
    const float tan_len_next = radii[i + 1] * tanf(angles[i + 1] / 2.0f);

    /* Scale down radii if too large for segment. */
    float factor = 1.0f;
    if (tan_len + tan_len_next > len_next) {
      factor = len_next / (tan_len + tan_len_next);
    }
    max_radii[i] = min_ff(max_radii[i], radii[i] * factor);
    max_radii[i + 1] = min_ff(max_radii[i + 1], radii[i + 1] * factor);
  }

  /* Assign the max_radii to the fillet data's radii. */
  for (const int i : IndexRange(size)) {
    radii[i] = max_radii[i];
  }
}

/*
 * Create a mapping from each vertex in the resulting spline to that of the source spline.
 * Used for copying the data from the source spline.
 */
static Array<int> create_dst_to_src_map(const Span<int> point_counts, const int total_points)
{
  Array<int> map(total_points);
  MutableSpan<int> map_span{map};
  int index = 0;

  for (const int i : point_counts.index_range()) {
    map_span.slice(index, point_counts[i]).fill(i);
    index += point_counts[i];
  }

  BLI_assert(index == total_points);

  return map;
}

/* Copy attribute data from source spline's Span to destination spline's Span. */
template<typename T>
static void copy_attribute_by_mapping(const Span<T> src,
                                      MutableSpan<T> dst,
                                      const Span<int> mapping)
{
  for (const int i : dst.index_range()) {
    dst[i] = src[mapping[i]];
  }
}

/* Copy all attributes in Bezier splines. */
static void copy_bezier_attributes_by_mapping(const BezierSpline &src,
                                              BezierSpline &dst,
                                              const Span<int> mapping)
{
  copy_attribute_by_mapping(src.positions(), dst.positions(), mapping);
  copy_attribute_by_mapping(src.radii(), dst.radii(), mapping);
  copy_attribute_by_mapping(src.tilts(), dst.tilts(), mapping);
  copy_attribute_by_mapping(src.handle_types_left(), dst.handle_types_left(), mapping);
  copy_attribute_by_mapping(src.handle_types_right(), dst.handle_types_right(), mapping);
  copy_attribute_by_mapping(src.handle_positions_left(), dst.handle_positions_left(), mapping);
  copy_attribute_by_mapping(src.handle_positions_right(), dst.handle_positions_right(), mapping);
}

/* Copy all attributes in Poly splines. */
static void copy_poly_attributes_by_mapping(const PolySpline &src,
                                            PolySpline &dst,
                                            const Span<int> mapping)
{
  copy_attribute_by_mapping(src.positions(), dst.positions(), mapping);
  copy_attribute_by_mapping(src.radii(), dst.radii(), mapping);
  copy_attribute_by_mapping(src.tilts(), dst.tilts(), mapping);
}

/* Copy all attributes in NURBS splines. */
static void copy_NURBS_attributes_by_mapping(const NURBSpline &src,
                                             NURBSpline &dst,
                                             const Span<int> mapping)
{
  copy_attribute_by_mapping(src.positions(), dst.positions(), mapping);
  copy_attribute_by_mapping(src.radii(), dst.radii(), mapping);
  copy_attribute_by_mapping(src.tilts(), dst.tilts(), mapping);
  copy_attribute_by_mapping(src.weights(), dst.weights(), mapping);
}

/* Update the positions and handle positions of a Bezier spline based on fillet data. */
static void update_bezier_positions(FilletData &fd,
                                    BezierSpline &dst_spline,
                                    const Span<int> point_counts,
                                    const int start,
                                    const int fillet_count)
{
  Span<float> radii(fd.radii);
  Span<float> angles(fd.angles);
  Span<float3> axes(fd.axes);
  Span<float3> positions(fd.positions);
  Span<float3> directions(fd.directions);

  const int size = radii.size();

  int cur_i = start;
  for (const int i : IndexRange(start, fillet_count)) {
    const int count = point_counts[i];

    /* Skip if the point count for the vertex is 1. */
    if (count == 1) {
      cur_i++;
      continue;
    }

    /* Calculate the angle to be formed between any 2 adjacent vertices within the fillet. */
    const float segment_angle = angles[i] / (count - 1);
    /* Calculate the handle length for each added vertex. Equation: L = 4R/3 * tan(A/4) */
    const float handle_length = 4.0f * radii[i] / 3.0f * tanf(segment_angle / 4.0f);
    /* Calculate the distance by which each vertex should be displaced from their initial position.
     */
    const float displacement = radii[i] * tanf(angles[i] / 2.0f);

    /* Position the end points of the arc and their handles. */
    const int end_i = cur_i + count - 1;
    const float3 prev_dir = i == 0 ? -directions[size - 1] : -directions[i - 1];
    const float3 next_dir = directions[i];
    dst_spline.positions()[cur_i] = positions[i] + displacement * prev_dir;
    dst_spline.positions()[end_i] = positions[i] + displacement * next_dir;
    dst_spline.handle_positions_right()[cur_i] = dst_spline.positions()[cur_i] -
                                                 handle_length * prev_dir;
    dst_spline.handle_positions_left()[cur_i] = dst_spline.positions()[cur_i] +
                                                handle_length * prev_dir;
    dst_spline.handle_positions_left()[end_i] = dst_spline.positions()[end_i] -
                                                handle_length * next_dir;
    dst_spline.handle_positions_right()[end_i] = dst_spline.positions()[end_i] +
                                                 handle_length * next_dir;
    dst_spline.handle_types_right()[cur_i] = dst_spline.handle_types_left()[end_i] =
        BezierSpline::HandleType::Align;
    dst_spline.handle_types_left()[cur_i] = dst_spline.handle_types_right()[end_i] =
        BezierSpline::HandleType::Vector;

    /* Calculate the center of the radius to be formed. */
    const float3 center = get_center(dst_spline.positions()[cur_i] - positions[i], fd, i);
    /* Calculate the vector of the radius formed by the first vertex. */
    float3 radius_vec = dst_spline.positions()[cur_i] - center;
    const float radius = radius_vec.normalize_and_get_length();

    /* For each of the vertices in between the end points. */
    for (const int j : IndexRange(1, count - 2)) {
      int index = cur_i + j;
      /* Rotate the radius by the segment angle and determine its tangent (used for getting handle
       * directions). */
      float3 new_radius_vec, tangent_vec;
      rotate_v3_v3v3fl(new_radius_vec, radius_vec, -axes[i], segment_angle);
      rotate_v3_v3v3fl(tangent_vec, new_radius_vec, axes[i], M_PI_2);
      radius_vec = new_radius_vec;
      tangent_vec *= handle_length;

      /* Adjust the positions of the respective vertex and its handles. */
      dst_spline.positions()[index] = center + new_radius_vec * radius;
      dst_spline.handle_types_right()[index] = dst_spline.handle_types_right()[index] =
          BezierSpline::HandleType::Align;
      dst_spline.handle_positions_left()[index] = dst_spline.positions()[index] + tangent_vec;
      dst_spline.handle_positions_right()[index] = dst_spline.positions()[index] - tangent_vec;
    }

    cur_i += count;
  }
}

/* Update the positions of a Poly spline based on fillet data. */
static void update_poly_or_NURBS_positions(FilletData &fd,
                                           Spline &dst_spline,
                                           const Span<int> point_counts,
                                           const int start,
                                           const int fillet_count)
{
  Span<float> radii(fd.radii);
  Span<float> angles(fd.angles);
  Span<float3> axes(fd.axes);
  Span<float3> positions(fd.positions);
  Span<float3> directions(fd.directions);

  const int size = radii.size();

  int cur_i = start;
  for (const int i : IndexRange(start, fillet_count)) {
    const int count = point_counts[i];

    /* Skip if the point count for the vertex is 1. */
    if (count == 1) {
      cur_i++;
      continue;
    }

    const float segment_angle = angles[i] / (count - 1);
    const float displacement = radii[i] * tanf(angles[i] / 2.0f);

    /* Position the end points of the arc. */
    const int end_i = cur_i + count - 1;
    const float3 prev_dir = i == 0 ? -directions[size - 1] : -directions[i - 1];
    const float3 next_dir = directions[i];
    dst_spline.positions()[cur_i] = positions[i] + displacement * prev_dir;
    dst_spline.positions()[end_i] = positions[i] + displacement * next_dir;

    /* Calculate the center of the radius to be formed. */
    const float3 center = get_center(dst_spline.positions()[cur_i] - positions[i], fd, i);
    /* Calculate the vector of the radius formed by the first vertex. */
    float3 radius_vec = dst_spline.positions()[cur_i] - center;

    for (const int j : IndexRange(1, count - 2)) {
      /* Rotate the radius by the segment angle */
      float3 new_radius_vec;
      rotate_v3_v3v3fl(new_radius_vec, radius_vec, -axes[i], segment_angle);
      radius_vec = new_radius_vec;

      dst_spline.positions()[cur_i + j] = center + new_radius_vec;
    }

    cur_i += count;
  }
}

/* Function to fillet a spline. */
static SplinePtr fillet_spline(const Spline &spline,
                               const FilletParam &fillet_param,
                               const int spline_index)
{
  int fillet_count, start = 0;
  const int size = spline.size();
  const bool cyclic = spline.is_cyclic();

  /* Determine the number of vertices that can be filleted. */
  if (cyclic) {
    fillet_count = size;
  }
  else {
    fillet_count = size - 2;
    start = 1;
  }

  if (size < 3) {
    return spline.copy();
  }

  /* Initialize the point_counts with 1s (at least one vertex on dst for each vertex on src). */
  Array<int> point_counts(size, 1.0f);

  int added_count = 0;
  /* Update point_counts array and added_count. */
  FilletData fd = calculate_fillet_data(
      spline, fillet_param, added_count, point_counts, spline_index);
  if (fillet_param.limit_radius) {
    limit_radii(fd, cyclic);
  }

  const int total_points = added_count + size;
  const Array<int> dst_to_src = create_dst_to_src_map(point_counts, total_points);
  SplinePtr dst_spline_ptr = spline.copy_only_settings();

  switch (spline.type()) {
    case Spline::Type::Bezier: {
      const BezierSpline &src_spline = static_cast<const BezierSpline &>(spline);
      BezierSpline &dst_spline = static_cast<BezierSpline &>(*dst_spline_ptr);
      dst_spline.resize(total_points);
      copy_bezier_attributes_by_mapping(src_spline, dst_spline, dst_to_src);
      if (fillet_param.mode == GEO_NODE_CURVE_FILLET_POLY) {
        dst_spline.handle_types_left().fill(BezierSpline::HandleType::Vector);
        dst_spline.handle_types_right().fill(BezierSpline::HandleType::Vector);
        update_poly_or_NURBS_positions(fd, dst_spline, point_counts, start, fillet_count);
      }
      else {
        update_bezier_positions(fd, dst_spline, point_counts, start, fillet_count);
      }
      break;
    }
    case Spline::Type::Poly: {
      const PolySpline &src_spline = static_cast<const PolySpline &>(spline);
      PolySpline &dst_spline = static_cast<PolySpline &>(*dst_spline_ptr);
      dst_spline.resize(total_points);
      copy_poly_attributes_by_mapping(src_spline, dst_spline, dst_to_src);
      update_poly_or_NURBS_positions(fd, dst_spline, point_counts, start, fillet_count);
      break;
    }
    case Spline::Type::NURBS: {
      const NURBSpline &src_spline = static_cast<const NURBSpline &>(spline);
      NURBSpline &dst_spline = static_cast<NURBSpline &>(*dst_spline_ptr);
      dst_spline.resize(total_points);
      copy_NURBS_attributes_by_mapping(src_spline, dst_spline, dst_to_src);
      update_poly_or_NURBS_positions(fd, dst_spline, point_counts, start, fillet_count);
      break;
    }
  }

  return dst_spline_ptr;
}

/* Function to fillet a curve */
static std::unique_ptr<CurveEval> fillet_curve(const CurveEval &input_curve,
                                               const FilletParam &fillet_param)
{
  Span<SplinePtr> input_splines = input_curve.splines();

  std::unique_ptr<CurveEval> output_curve = std::make_unique<CurveEval>();
  const int num_splines = input_splines.size();
  output_curve->resize(num_splines);
  MutableSpan<SplinePtr> output_splines = output_curve->splines();

  Array<int> spline_indices(input_splines.size());
  spline_indices[0] = 0;
  for (const int i : IndexRange(1, num_splines - 1)) {
    spline_indices[i] = spline_indices[i - 1] + input_splines[i - 1]->size();
  }

  threading::parallel_for(input_splines.index_range(), 128, [&](IndexRange range) {
    for (const int i : range) {
      output_splines[i] = fillet_spline(*input_splines[i], fillet_param, spline_indices[i]);
    }
  });

  return output_curve;
}

static void geo_node_fillet_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Curve");

  geometry_set = bke::geometry_set_realize_instances(geometry_set);

  if (!geometry_set.has_curve()) {
    params.set_output("Curve", geometry_set);
    return;
  }

  const CurveEval &input_curve = *geometry_set.get_curve_for_read();
  NodeGeometryCurveFillet &node_storage = *(NodeGeometryCurveFillet *)params.node().storage;
  const GeometryNodeCurveFilletMode mode = (GeometryNodeCurveFilletMode)node_storage.mode;
  const GeometryNodeAttributeInputMode radius_mode = (GeometryNodeAttributeInputMode)
                                                         node_storage.radius_mode;
  FilletParam fillet_param;
  fillet_param.mode = mode;

  if (mode == GEO_NODE_CURVE_FILLET_POLY) {
    const int count = params.extract_input<int>("Poly Count");
    if (count < 1) {
      params.set_output("Curve", GeometrySet());
      return;
    }
    fillet_param.count.emplace(count);
  }
  else {
    fillet_param.count.emplace(1);
  }

  fillet_param.limit_radius = params.extract_input<bool>("Limit Radius");

  std::unique_ptr<CurveEval> output_curve;
  GVArray_Typed<float> radii_array = params.get_input_attribute<float>(
      "Radius", *geometry_set.get_component_for_read<CurveComponent>(), ATTR_DOMAIN_POINT, 0.0f);

  if (radii_array->is_single() && radii_array->get_internal_single() < 0) {
    params.set_output("Geometry", geometry_set);
    return;
  }

  fillet_param.radii = &radii_array;
  output_curve = fillet_curve(input_curve, fillet_param);

  params.set_output("Curve", GeometrySet::create_with_curve(output_curve.release()));
}
}  // namespace blender::nodes

void register_node_type_geo_curve_fillet()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_CURVE_FILLET, "Curve Fillet", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, geo_node_curve_fillet_in, geo_node_curve_fillet_out);
  ntype.draw_buttons = geo_node_curve_fillet_layout;
  node_type_storage(
      &ntype, "NodeGeometryCurveFillet", node_free_standard_storage, node_copy_standard_storage);
  node_type_init(&ntype, geo_node_curve_fillet_init);
  node_type_update(&ntype, blender::nodes::geo_node_curve_fillet_update);
  ntype.geometry_node_execute = blender::nodes::geo_node_fillet_exec;
  nodeRegisterType(&ntype);
}
