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

#include "node_geometry_util.hh"

#include "BKE_spline.hh"

#include "float.h"

static bNodeSocketTemplate geo_node_curve_fillet_in[] = {
    {SOCK_GEOMETRY, N_("Curve")},
    {SOCK_FLOAT, N_("Angle"), M_PI_2, 0.0f, 0.0f, 0.0f, 0.001f, FLT_MAX, PROP_ANGLE},
    {SOCK_INT, N_("Count"), 1, 0, 0, 0, 1, 1000},
    {SOCK_BOOLEAN, N_("Limit Radius")},
    {SOCK_FLOAT, N_("Radius"), 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, FLT_MAX, PROP_DISTANCE},
    {SOCK_STRING, N_("Radii")},
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

  data->mode = GEO_NODE_CURVE_FILLET_ADAPTIVE;
  data->radius_mode = GEO_NODE_CURVE_FILLET_RADIUS_FLOAT;

  node->storage = data;
}

namespace blender::nodes {

struct FilletModeParam {
  GeometryNodeCurveFilletMode mode{};

  /* Minimum angle between two adjust control points. */
  std::optional<float> angle;

  /* Number of points to be added. */
  std::optional<int> count;

  GeometryNodeCurveFilletRadiusMode radius_mode{};

  /* Whether or not fillets are allowed to overlap. */
  bool limit_radius;

  /* The radius of the formed circle */
  std::optional<float> radius;

  /* Distribution of radii on the spline. */
  std::optional<std::string> radii_dist;

  GVArray_Typed<float> *radii{};
};

/* A data structure used to store fillet data about a vertex in the source spline. */
struct FilletData {
  float3 prev_dir, pos, next_dir, axis;
  float radius, angle, count;
};

static void geo_node_curve_fillet_update(bNodeTree *UNUSED(ntree), bNode *node)
{
  NodeGeometryCurveFillet &node_storage = *(NodeGeometryCurveFillet *)node->storage;
  const GeometryNodeCurveFilletMode mode = (GeometryNodeCurveFilletMode)node_storage.mode;

  bNodeSocket *adaptive_socket = ((bNodeSocket *)node->inputs.first)->next;
  bNodeSocket *user_socket = adaptive_socket->next;

  nodeSetSocketAvailability(adaptive_socket, mode == GEO_NODE_CURVE_FILLET_ADAPTIVE);
  nodeSetSocketAvailability(user_socket, mode == GEO_NODE_CURVE_FILLET_USER_DEFINED);

  const GeometryNodeCurveFilletMode radius_mode = (GeometryNodeCurveFilletMode)
                                                      node_storage.radius_mode;

  bNodeSocket *float_socket = user_socket->next->next;
  bNodeSocket *attribute_socket = float_socket->next;

  nodeSetSocketAvailability(float_socket, radius_mode == GEO_NODE_CURVE_FILLET_RADIUS_FLOAT);
  nodeSetSocketAvailability(attribute_socket,
                            radius_mode == GEO_NODE_CURVE_FILLET_RADIUS_ATTRIBUTE);
}

/* Function to get the center of a fillet. */
static float3 get_center(const float3 vec_pos2prev,
                         const float3 pos,
                         const float3 axis,
                         const float angle)
{
  float3 vec_pos2center;
  rotate_v3_v3v3fl(vec_pos2center, vec_pos2prev, axis, M_PI_2 - angle / 2);
  vec_pos2center *= 1 / sinf(angle / 2);

  return vec_pos2center + pos;
}

/* Function to get the center of the fillet using fillet data */
static float3 get_center(const float3 vec_pos2prev, const FilletData &fd)
{
  float angle = fd.angle;
  float3 axis = fd.axis;
  float3 pos = fd.pos;

  return get_center(vec_pos2prev, pos, axis, angle);
}

/* Function to calculate fillet data for each vertex. */
static FilletData calculate_fillet_data_per_vertex(const float3 prev_pos,
                                                   const float3 pos,
                                                   const float3 next_pos,
                                                   const std::optional<float> arc_angle,
                                                   const std::optional<int> count,
                                                   const float radius,
                                                   const bool limit_radius)
{
  FilletData fd{};
  float3 vec_pos2prev = prev_pos - pos;
  float3 vec_pos2next = next_pos - pos;
  normalize_v3_v3(fd.prev_dir, vec_pos2prev);
  normalize_v3_v3(fd.next_dir, vec_pos2next);
  fd.pos = pos;
  cross_v3_v3v3(fd.axis, vec_pos2prev, vec_pos2next);
  fd.angle = M_PI - angle_v3v3v3(prev_pos, pos, next_pos);
  fd.count = count.has_value() ? count.value() : fd.angle / arc_angle.value();
  fd.radius = radius;

  return fd;
}

/* Limit the radius based on angle and radii to prevent overlap. */
static void limit_radii(Array<FilletData> &fds,
                        const Span<float3> &positions,
                        const int size,
                        const bool cyclic)
{
  int fillet_count, start = 0;
  Array<float> max_radii(size, {-1});

  /* Handle the corner cases if cyclic. */
  if (cyclic) {
    fillet_count = size;

    float len_prev, len_next, tan_len, tan_len_prev, tan_len_next;

    /* Calculate lengths between adjacent control points. */
    len_prev = len_v3v3(fds[0].pos, fds[size - 1].pos);
    len_next = len_v3v3(fds[0].pos, fds[1].pos);

    /* Calculate tangent lengths of fillets in control points. */
    tan_len = fds[0].radius * tanf(fds[0].angle / 2);
    tan_len_prev = fds[size - 1].radius * tanf(fds[size - 1].angle / 2);
    tan_len_next = fds[1].radius * tanf(fds[1].angle / 2);

    max_radii[0] = fds[0].radius;
    max_radii[1] = fds[1].radius;
    max_radii[size - 1] = fds[size - 1].radius;

    float factor_prev = 1, factor_next = 1;
    if (tan_len + tan_len_prev > len_prev) {
      factor_prev = len_prev / (tan_len + tan_len_prev);
    }
    if (tan_len + tan_len_next > len_next) {
      factor_next = len_next / (tan_len + tan_len_next);
    }

    /* Scale max radii by calculated factors. */
    max_radii[0] *= min_ff(factor_next, factor_prev);
    max_radii[1] *= factor_next;
    max_radii[size - 1] *= factor_prev;
  }
  else {
    fillet_count = size - 2;
    start = 1;
  }

  /* Initialize max_radii to largest possible radii. */
  for (const int i : IndexRange(start, fillet_count)) {
    if (i > 0 && i < size - 1) {
      max_radii[i] = min_ff(len_v3v3(positions[i], positions[i - 1]),
                            len_v3v3(positions[i], positions[i + 1])) /
                     tanf(fds[i - start].angle / 2);
    }
  }

  /* Max radii calculations for each index. */
  for (const int i : IndexRange(start, fillet_count - 1)) {
    int fillet_i = i - start;
    float len_next = len_v3v3(fds[fillet_i].pos, fds[fillet_i + 1].pos);
    float tan_len = fds[fillet_i].radius * tanf(fds[fillet_i].angle / 2);
    float tan_len_next = fds[fillet_i + 1].radius * tanf(fds[fillet_i + 1].angle / 2);

    /* Scale down radii if too large for segment. */
    float factor = 1;
    if (tan_len + tan_len_next > len_next) {
      factor = len_next / (tan_len + tan_len_next);
    }
    max_radii[i] = min_ff(max_radii[i], fds[fillet_i].radius * factor);
    max_radii[i + 1] = min_ff(max_radii[i + 1], fds[fillet_i + 1].radius * factor);
  }

  /* Assign the max_radii to the fillet data's radii. */
  for (const int i : IndexRange(fillet_count)) {
    fds[i].radius = max_radii[start + i];
  }
}

/* Function to calculate and obtain the fillet data for the entire spline. */
static Array<FilletData> calculate_fillet_data(const SplinePtr &spline,
                                               const FilletModeParam &mode_param,
                                               int &added_count,
                                               Array<int> &point_counts)
{
  Span<float3> positions = spline->positions();
  int fillet_count, start = 0, size = spline->size();

  /* Determine the number of vertices that can be filleted. */
  bool cyclic = spline->is_cyclic();
  if (!cyclic) {
    fillet_count = size - 2;
    start = 1;
  }
  else {
    fillet_count = size;
  }

  Array<FilletData> fds(fillet_count);

  for (const int i : IndexRange(fillet_count)) {
    /* Find the positions of the adjacent vertices. */
    float3 prev_pos, pos, next_pos;
    if (cyclic) {
      prev_pos = positions[i == 0 ? positions.size() - 1 : i - 1];
      pos = positions[i];
      next_pos = positions[i == positions.size() - 1 ? 0 : i + 1];
    }
    else {
      prev_pos = positions[i];
      pos = positions[i + 1];
      next_pos = positions[i + 2];
    }

    /* Define the radius. */
    float radius = 0.0f;
    if (mode_param.radius_mode == GEO_NODE_CURVE_FILLET_RADIUS_FLOAT) {
      radius = mode_param.radius.value();
    }
    else if (mode_param.radius_mode == GEO_NODE_CURVE_FILLET_RADIUS_ATTRIBUTE) {
      if (mode_param.radii->size() != size) {
        radius = 0;
      }
      else {
        radius = (*mode_param.radii)[start + i];
      }
    }

    /* Calculate fillet data for the vertex. */
    fds[i] = calculate_fillet_data_per_vertex(prev_pos,
                                              pos,
                                              next_pos,
                                              mode_param.angle,
                                              mode_param.count,
                                              radius,
                                              mode_param.limit_radius);

    /* Exit from here if the radius is zero */
    if (!radius) {
      continue;
    }

    /* Calculate number of points to be added for the vertex. */
    int count = 0;
    if (mode_param.mode == GEO_NODE_CURVE_FILLET_ADAPTIVE) {
      count = ceil(fds[i].angle / mode_param.angle.value());
    }
    else if (mode_param.mode == GEO_NODE_CURVE_FILLET_USER_DEFINED) {
      count = mode_param.count.value();
    }

    added_count += count;
    point_counts[start + i] = count + 1;
  }

  return fds;
}

/*
 * Create a mapping from each vertex in the resulting spline to that of the source spline.
 * Used for copying the data from the source spline.
 */
static Array<int> create_dst_to_src_map(const Array<int> point_counts, const int total_points)
{
  Array<int> map(total_points);
  int index = 0;

  for (int i = 0; i < point_counts.size(); i++) {
    for (int j = 0; j < point_counts[i]; j++) {
      map[index] = i;
      index++;
    }
  }

  BLI_assert(index == total_points);

  return map;
}

/* Copy attribute data from source spline's Span to destination spline's Span. */
template<typename T>
static void copy_attribute_by_mapping(const Span<T> src, MutableSpan<T> dst, Array<int> mapping)
{
  for (const int i : dst.index_range()) {
    dst[i] = src[mapping[i]];
  }
}

/* Copy all attributes in Bezier splines. */
static void copy_bezier_attributes_by_mapping(const BezierSpline &src,
                                              BezierSpline &dst,
                                              Array<int> mapping)
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
                                            const Array<int> mapping)
{
  copy_attribute_by_mapping(src.positions(), dst.positions(), mapping);
  copy_attribute_by_mapping(src.radii(), dst.radii(), mapping);
  copy_attribute_by_mapping(src.tilts(), dst.tilts(), mapping);
}

/* Copy all attributes in NURBS splines. */
static void copy_NURBS_attributes_by_mapping(const NURBSpline &src,
                                             NURBSpline &dst,
                                             const Array<int> mapping)
{
  copy_attribute_by_mapping(src.positions(), dst.positions(), mapping);
  copy_attribute_by_mapping(src.radii(), dst.radii(), mapping);
  copy_attribute_by_mapping(src.tilts(), dst.tilts(), mapping);
  copy_attribute_by_mapping(src.weights(), dst.weights(), mapping);
}

/*
 * Update the handle types to get a smoother curve.
 * Required when handles are not of type `Vector`.
 */
static void update_bezier_handle_types(BezierSpline &dst,
                                       const Array<int> mapping,
                                       const Array<int> point_counts)
{
  MutableSpan<BezierSpline::HandleType> left_handle_types = dst.handle_types_left();
  MutableSpan<BezierSpline::HandleType> right_handle_types = dst.handle_types_right();
  bool is_cyclic = dst.is_cyclic();
  for (int i = 0; i < mapping.size(); i++) {
    if (point_counts[mapping[i]] > 1) {
      /* There will always be a prev and next if point count > 1. */
      int prev = 0, next = 0;
      if (i == 0) {
        prev = mapping.size() - 1;
        next = i + 1;
      }
      else if (i < mapping.size() - 1) {
        prev = i - 1;
        next = i + 1;
      }
      else {
        prev = i - 1;
        next = 0;
      }

      /* Update handle types of adjacent points. */
      if (mapping[i] != mapping[prev]) {
        right_handle_types[prev] = BezierSpline::HandleType::Vector;
      }
      if (mapping[i] != mapping[next]) {
        left_handle_types[next] = BezierSpline::HandleType::Vector;
      }
    }
  }
}

/* Update the positions and handle positions of a Bezier spline based on fillet data. */
static void update_bezier_positions(Array<FilletData> &fds,
                                    BezierSpline &dst_spline,
                                    const Array<int> point_counts,
                                    const int start,
                                    const int fillet_count)
{
  int cur_i = start;
  for (const int i : IndexRange(start, fillet_count)) {
    FilletData fd = fds[i - start];
    int count = point_counts[i];

    /* Skip if the point count for the vertex is 1. */
    if (count == 1) {
      cur_i++;
      continue;
    }

    /* Calculate the angle to be formed between any 2 adjacent vertices within the fillet. */
    float segment_angle = fd.angle / (count - 1);
    /* Calculate the handle length for each added vertex. Equation: L = 4R/3 * tan(A/4) */
    float handle_length = 4.0f * fd.radius / 3 * tanf(segment_angle / 4);
    /* Calculate the distance by which each vertex should be displaced from their initial position.
     */
    float displacement = fd.radius * tanf(fd.angle / 2);

    /* Position the end points of the arc and their handles. */
    int end_i = cur_i + count - 1;
    dst_spline.positions()[cur_i] = fd.pos + displacement * fd.prev_dir;
    dst_spline.positions()[end_i] = fd.pos + displacement * fd.next_dir;
    dst_spline.handle_positions_right()[cur_i] = dst_spline.positions()[cur_i] -
                                                 handle_length * fd.prev_dir;
    dst_spline.handle_positions_left()[end_i] = dst_spline.positions()[end_i] -
                                                handle_length * fd.next_dir;
    dst_spline.handle_types_right()[cur_i] = dst_spline.handle_types_left()[end_i] =
        BezierSpline::HandleType::Align;
    dst_spline.handle_types_left()[cur_i] = dst_spline.handle_types_right()[end_i] =
        BezierSpline::HandleType::Vector;

    /* Calculate the center of the radius to be formed. */
    float3 center = get_center(dst_spline.positions()[cur_i] - fd.pos, fd);
    /* Calculate the vector of the radius formed by the first vertex. */
    float3 radius_vec = dst_spline.positions()[cur_i] - center;

    /* For each of the vertices in between the end points. */
    for (int j = 1; j < count - 1; j++) {
      int index = cur_i + j;
      /* Rotate the radius by the segment angle and determine its tangent (used for getting handle
       * directions). */
      float3 new_radius_vec, tangent_vec;
      rotate_v3_v3v3fl(new_radius_vec, radius_vec, -fd.axis, segment_angle);
      rotate_v3_v3v3fl(tangent_vec, new_radius_vec, fd.axis, M_PI_2);
      radius_vec = new_radius_vec;
      normalize_v3_length(tangent_vec, handle_length);

      /* Adjust the positions of the respective vertex and its handles. */
      dst_spline.positions()[index] = center + new_radius_vec;
      dst_spline.handle_types_right()[index] = dst_spline.handle_types_right()[index] =
          BezierSpline::HandleType::Align;
      dst_spline.handle_positions_left()[index] = dst_spline.positions()[index] + tangent_vec;
      dst_spline.handle_positions_right()[index] = dst_spline.positions()[index] - tangent_vec;
    }

    cur_i += count;
  }
}

/* Update the positions of a Poly spline based on fillet data. */
static void update_poly_or_NURBS_positions(Array<FilletData> &fds,
                                           Spline &dst_spline,
                                           const Array<int> point_counts,
                                           const int start,
                                           const int fillet_count)
{
  int cur_i = start;
  for (const int i : IndexRange(start, fillet_count)) {
    FilletData fd = fds[i - start];
    int count = point_counts[i];

    /* Skip if the point count for the vertex is 1. */
    if (count == 1) {
      cur_i++;
      continue;
    }

    float segment_angle = fd.angle / (count - 1);
    float displacement = fd.radius * tanf(fd.angle / 2);

    /* Position the end points of the arc. */
    int end_i = cur_i + count - 1;
    dst_spline.positions()[cur_i] = fd.pos + displacement * fd.prev_dir;
    dst_spline.positions()[end_i] = fd.pos + displacement * fd.next_dir;

    /* Calculate the center of the radius to be formed. */
    float3 center = get_center(dst_spline.positions()[cur_i] - fd.pos, fd);
    /* Calculate the vector of the radius formed by the first vertex. */
    float3 radius_vec = dst_spline.positions()[cur_i] - center;

    for (int j = 1; j < count - 1; j++) {
      /* Rotate the radius by the segment angle */
      float3 new_radius_vec;
      rotate_v3_v3v3fl(new_radius_vec, radius_vec, -fd.axis, segment_angle);
      radius_vec = new_radius_vec;

      dst_spline.positions()[cur_i + j] = center + new_radius_vec;
    }

    cur_i += count;
  }
}

/* Function to fillet a spline. */
static SplinePtr fillet_spline(const Spline &spline, const FilletModeParam &mode_param)
{
  int fillet_count, start = 0, size = spline.size();
  bool cyclic = spline.is_cyclic();
  SplinePtr src_spline_ptr = spline.copy();

  /* Determine the number of vertices that can be filleted. */
  if (!cyclic) {
    fillet_count = size - 2;
    start = 1;
  }
  else {
    fillet_count = size;
  }

  if (fillet_count <= 0) {
    return src_spline_ptr;
  }

  /* Initialize the point_counts with 1s (at least one vertex on dst for each vertex on src). */
  Array<int> point_counts(size, {1});

  int added_count = 0;
  /* Update point_counts array and added_count. */
  Array<FilletData> fds = calculate_fillet_data(
      src_spline_ptr, mode_param, added_count, point_counts);
  if (mode_param.limit_radius) {
    limit_radii(fds, spline.positions(), spline.size(), cyclic);
  }

  int total_points = added_count + size;
  Array<int> dst_to_src = create_dst_to_src_map(point_counts, total_points);
  SplinePtr dst_spline_ptr = spline.copy_only_settings();

  switch (spline.type()) {
    case Spline::Type::Bezier: {
      BezierSpline src_spline = static_cast<BezierSpline &>(*src_spline_ptr);
      BezierSpline &dst_spline = static_cast<BezierSpline &>(*dst_spline_ptr);
      dst_spline.resize(total_points);
      copy_bezier_attributes_by_mapping(src_spline, dst_spline, dst_to_src);
      update_bezier_handle_types(dst_spline, dst_to_src, point_counts);
      update_bezier_positions(fds, dst_spline, point_counts, start, fillet_count);
      break;
    }
    case Spline::Type::Poly: {
      PolySpline src_spline = static_cast<PolySpline &>(*src_spline_ptr);
      PolySpline &dst_spline = static_cast<PolySpline &>(*dst_spline_ptr);
      dst_spline.resize(total_points);
      copy_poly_attributes_by_mapping(src_spline, dst_spline, dst_to_src);
      update_poly_or_NURBS_positions(fds, dst_spline, point_counts, start, fillet_count);
      break;
    }
    case Spline::Type::NURBS: {
      NURBSpline src_spline = static_cast<NURBSpline &>(*src_spline_ptr);
      NURBSpline &dst_spline = static_cast<NURBSpline &>(*dst_spline_ptr);
      dst_spline.resize(total_points);
      copy_NURBS_attributes_by_mapping(src_spline, dst_spline, dst_to_src);
      update_poly_or_NURBS_positions(fds, dst_spline, point_counts, start, fillet_count);
      break;
    }
  }

  return dst_spline_ptr;
}

/* Function to fillet a curve */
static std::unique_ptr<CurveEval> fillet_curve(const CurveEval &input_curve,
                                               const FilletModeParam &mode_param)
{
  Span<SplinePtr> input_splines = input_curve.splines();

  std::unique_ptr<CurveEval> output_curve = std::make_unique<CurveEval>();
  output_curve->resize(input_splines.size());
  MutableSpan<SplinePtr> output_splines = output_curve->splines();

  if (mode_param.radius_mode == GEO_NODE_CURVE_FILLET_RADIUS_ATTRIBUTE) {
    threading::parallel_for(input_splines.index_range(), 128, [&](IndexRange range) {
      for (const int i : range) {
        const Spline &spline = *input_splines[i];
        std::string radii_name = mode_param.radii_dist.value();
        GVArray_Typed<float> radii_dist = spline.attributes.get_for_read<float>(radii_name, 1.0f);
        output_splines[i] = fillet_spline(spline, mode_param);
      }
    });
  }
  else {
    threading::parallel_for(input_splines.index_range(), 128, [&](IndexRange range) {
      for (const int i : range) {
        output_splines[i] = fillet_spline(*input_splines[i], mode_param);
      }
    });
  }

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
  const GeometryNodeCurveFilletRadiusMode radius_mode = (GeometryNodeCurveFilletRadiusMode)
                                                            node_storage.radius_mode;
  FilletModeParam mode_param;
  mode_param.mode = mode;
  mode_param.radius_mode = radius_mode;

  if (mode == GEO_NODE_CURVE_FILLET_ADAPTIVE) {
    const float angle = std::max(params.extract_input<float>("Angle"), 0.0001f);
    mode_param.angle.emplace(angle);
  }
  else if (mode == GEO_NODE_CURVE_FILLET_USER_DEFINED) {
    const int count = params.extract_input<int>("Count");
    if (count < 1) {
      params.set_output("Curve", GeometrySet());
      return;
    }
    mode_param.count.emplace(count);
  }

  mode_param.limit_radius = params.extract_input<bool>("Limit Radius");

  if (radius_mode == GEO_NODE_CURVE_FILLET_RADIUS_FLOAT) {
    mode_param.radius.emplace(params.extract_input<float>("Radius"));
  }
  else {
    GVArray_Typed<float> radii_array = params.get_input_attribute<float>(
        "Radii", geometry_set.get_component_for_write<CurveComponent>(), ATTR_DOMAIN_AUTO, 0.0f);

    mode_param.radii = &radii_array;
    mode_param.radii_dist.emplace(params.extract_input<std::string>("Radii"));
  }

  std::unique_ptr<CurveEval> output_curve = fillet_curve(input_curve, mode_param);

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
