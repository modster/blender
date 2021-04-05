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
#include "BLI_listbase.h"
#include "BLI_span.hh"

#include "DNA_curve_types.h"

#include "BKE_curve.h"
#include "BKE_derived_curve.hh"

using blender::Array;
using blender::float3;
using blender::IndexRange;
using blender::MutableSpan;
using blender::Span;

/** \} */

/* -------------------------------------------------------------------- */
/** \name General Curve Functions
 * \{ */

static BezierPoint::HandleType handle_type_from_dna_bezt(const eBezTriple_Handle dna_handle_type)
{
  switch (dna_handle_type) {
    case HD_FREE:
      return BezierPoint::Free;
    case HD_AUTO:
      return BezierPoint::Auto;
    case HD_VECT:
      return BezierPoint::Vector;
    case HD_ALIGN:
      return BezierPoint::Align;
    case HD_AUTO_ANIM:
      return BezierPoint::Auto;
    case HD_ALIGN_DOUBLESIDE:
      return BezierPoint::Align;
  }
  BLI_assert_unreachable();
  return BezierPoint::Free;
}

DCurve *dcurve_from_dna_curve(const Curve &dna_curve)
{
  DCurve *curve = new DCurve();

  const ListBase *nurbs = BKE_curve_nurbs_get(&const_cast<Curve &>(dna_curve));

  curve->splines.reserve(BLI_listbase_count(nurbs));

  LISTBASE_FOREACH (const Nurb *, nurb, nurbs) {
    if (nurb->type == CU_BEZIER) {
      BezierSpline *spline = new BezierSpline();
      for (const BezTriple &bezt : Span(nurb->bezt, nurb->pntsu)) {
        BezierPoint point;
        point.handle_position_a = bezt.vec[0];
        point.position = bezt.vec[1];
        point.handle_position_b = bezt.vec[2];
        point.radius = bezt.radius;
        point.tilt = bezt.tilt;
        point.handle_type_a = handle_type_from_dna_bezt((eBezTriple_Handle)bezt.h1);
        point.handle_type_b = handle_type_from_dna_bezt((eBezTriple_Handle)bezt.h2);
        spline->control_points.append(std::move(point));
      }

      spline->resolution_u = nurb->resolu;
      // spline.resolution_v = nurb->resolv;
      spline->type = Spline::Type::Bezier;

      spline->is_cyclic = nurb->flagu & CU_NURB_CYCLIC;

      curve->splines.append(spline);
    }
    else if (nurb->type == CU_NURBS) {
    }
    else if (nurb->type == CU_POLY) {
    }
  }

  return curve;
}

/* -------------------------------------------------------------------- */
/** \name Bezier Spline
 * \{ */

static bool segment_is_vector(const BezierPoint &point_a, const BezierPoint &point_b)
{
  return point_a.handle_type_b == BezierPoint::HandleType::Vector &&
         point_b.handle_type_a == BezierPoint::HandleType::Vector;
}

int BezierSpline::evaluated_points_size() const
{
  BLI_assert(control_points.size() > 0);

  int total_len = 0;
  for (const int i : IndexRange(1, this->control_points.size() - 1)) {
    const BezierPoint &point_prev = this->control_points[i - 1];
    const BezierPoint &point = this->control_points[i];
    if (segment_is_vector(point_prev, point)) {
      total_len += 1;
    }
    else {
      total_len += this->resolution_u;
    }
  }

  if (this->is_cyclic) {
    if (segment_is_vector(this->control_points.last(), this->control_points.first())) {
      total_len++;
    }
    else {
      total_len += this->resolution_u;
    }
  }
  else {
    /* Since evaulating the bezier doesn't add the final point's position,
     * it must be added manually in the non-cyclic case. */
    total_len++;
  }

  return total_len;
}

static void evaluate_bezier_part_3d(const float3 &point_0,
                                    const float3 &point_1,
                                    const float3 &point_2,
                                    const float3 &point_3,
                                    MutableSpan<float3> result)
{
  /* TODO: This can probably be vectorized... no one has done this already? */
  float *data = (float *)result.data();
  for (const int axis : {0, 1, 2}) {
    BKE_curve_forward_diff_bezier(point_0[axis],
                                  point_1[axis],
                                  point_2[axis],
                                  point_3[axis],
                                  data + axis,
                                  result.size(),
                                  sizeof(float3));
  }
}

static void evaluate_segment_positions(const BezierPoint &point,
                                       const BezierPoint &next,
                                       const int resolution,
                                       int &offset,
                                       MutableSpan<float3> positions)
{
  if (segment_is_vector(point, next)) {
    positions[offset] = point.position;
    offset++;
  }
  else {
    evaluate_bezier_part_3d(point.position,
                            point.handle_position_b,
                            next.handle_position_a,
                            next.position,
                            positions.slice(offset, resolution - 1));
    offset += resolution;
  }
}

static void evaluate_positions(Span<BezierPoint> control_points,
                               const int resolution,
                               const bool is_cyclic,
                               MutableSpan<float3> positions)
{
  int offset = 0;
  for (const int i : IndexRange(1, control_points.size() - 1)) {
    const BezierPoint &point_prev = control_points[i - 1];
    const BezierPoint &point = control_points[i];
    evaluate_segment_positions(point_prev, point, resolution, offset, positions);
  }

  if (is_cyclic) {
    const BezierPoint &last_point = control_points.last();
    const BezierPoint &first_point = control_points.first();
    evaluate_segment_positions(last_point, first_point, resolution, offset, positions);
  }
  else {
    /* Since evaulating the bezier doesn't add the final point's position,
     * it must be added manually in the non-cyclic case. */
    positions[offset] = control_points.last().position;
    offset++;
  }

  BLI_assert(offset == positions.size());
}

static float3 direction_bisect(const float3 &prev, const float3 &middle, const float3 &next)
{
  const float3 dir_prev = (middle - prev).normalized();
  const float3 dir_next = (next - middle).normalized();

  return (dir_prev + dir_next).normalized();
}

static void calculate_tangents(Span<BezierPoint> control_points,
                               Span<float3> positions,
                               const bool is_cyclic,
                               MutableSpan<float3> tangents)
{
  if (positions.size() == 1) {
    tangents.first() = float3(0.0f, 0.0f, 0.0f);
    return;
  }

  for (const int i : IndexRange(1, positions.size() - 2)) {
    tangents[i] = direction_bisect(positions[i - 1], positions[i], positions[i + 1]);
  }

  if (is_cyclic) {
    const float3 &second_to_last = positions[positions.size() - 2];
    const float3 &last = positions.last();
    const float3 &first = positions.first();
    const float3 &second = positions[1];
    tangents.first() = direction_bisect(last, first, second);
    tangents.last() = direction_bisect(second_to_last, last, first);
  }
  else {
    /* If the spline is not cyclic, the direction for the first and last points is just the
     * direction formed by the corresponding handles and control points. In the unlikely situation
     * that the handles define a zero direction, fallback to using the direction defined by the
     * first and last evaluated segments. */
    const BezierPoint &first_point = control_points.first();
    if (LIKELY(first_point.handle_position_a != first_point.position)) {
      tangents.first() = (first_point.position - first_point.handle_position_a).normalized();
    }
    else {
      tangents.first() = (positions[1] - positions[0]).normalized();
    }

    const BezierPoint &last_point = control_points.last();
    if (LIKELY(last_point.handle_position_a != first_point.position)) {
      tangents.last() = (last_point.handle_position_b - last_point.position).normalized();
    }
    else {
      tangents.last() = (positions.last() - positions[positions.size() - 1]).normalized();
    }
  }

  for (const float3 &tangent : tangents) {
    BLI_ASSERT_UNIT_V3(tangent);
  }
}

static float3 initial_normal(const float3 first_tangent)
{
  /* TODO: Should be is "almost" zero. */
  if (first_tangent.is_zero()) {
    return float3(0.0f, 0.0f, 1.0f);
  }

  const float3 normal = float3::cross(first_tangent, float3(0.0f, 0.0f, 1.0f));
  if (!normal.is_zero()) {
    return normal.normalized();
  }

  return float3::cross(first_tangent, float3(0.0f, 1.0f, 0.0f)).normalized();
}

static float3 rotate_around_axis(const float3 dir, const float3 axis, const float angle)
{
  // cdef void rotateAroundAxisVec3(Vector3 *target, Vector3 *v, Vector3 *axis, float angle):
  //   cdef Vector3 n
  //   normalizeVec3(&n, axis)
  //   cdef Vector3 d
  //   scaleVec3(&d, &n, dotVec3(&n, v))
  //   cdef Vector3 r
  //   subVec3(&r, v, &d)
  //   cdef Vector3 g
  //   crossVec3(&g, &n, &r)
  //   cdef float ca = cos(angle)
  //   cdef float sa = sin(angle)
  //   target.x = d.x + r.x * ca + g.x * sa
  //   target.y = d.y + r.y * ca + g.y * sa
  //   target.z = d.z + r.z * ca + g.z * sa
  BLI_ASSERT_UNIT_V3(axis);
  const float3 scaled_axis = axis * float3::dot(dir, axis);
  const float3 sub = dir - scaled_axis;
  const float3 cross = float3::cross(sub, sub);
  const float sin = std::sin(angle);
  const float cos = std::cos(angle);
  return (scaled_axis + sub * cos + cross * sin).normalized();
}

static float3 project_on_center_plane(const float3 vector, const float3 plane_normal)
{
  // cdef void projectOnCenterPlaneVec3(Vector3 *result, Vector3 *v, Vector3 *planeNormal):
  //   cdef Vector3 unitNormal, projVector
  //   normalizeVec3(&unitNormal, planeNormal)
  //   cdef float distance = dotVec3(v, &unitNormal)
  //   scaleVec3(&projVector, &unitNormal, -distance)
  //   addVec3(result, v, &projVector)
  BLI_ASSERT_UNIT_V3(plane_normal);
  const float distance = float3::dot(vector, plane_normal);
  const float3 projection_vector = plane_normal * -distance;
  return vector + projection_vector;
}

static float3 propagate_normal(const float3 last_normal,
                               const float3 last_tangent,
                               const float3 current_tangent)
{
  const float angle = angle_normalized_v3v3(last_tangent, current_tangent);

  if (angle == 0.0f) {
    return last_normal;
  }

  const float3 axis = float3::cross(last_tangent, current_tangent).normalized();

  // rotateAroundAxisVec3(&newNormal, lastNormal, &axis, angle)
  // projectOnCenterPlaneVec3(target, &newNormal, currentTangent)
  const float3 new_normal = rotate_around_axis(last_normal, axis, angle);

  return project_on_center_plane(new_normal, current_tangent).normalized();
}

static void apply_rotation_gradient(Span<float3> tangents,
                                    MutableSpan<float3> normals,
                                    const float full_angle)
{
  // cdef applyRotationGradient(Vector3DList tangents, Vector3DList normals, float fullAngle):
  //   cdef Py_ssize_t i
  //   cdef Vector3 normal
  //   cdef float angle
  //   cdef float remainingRotation = fullAngle
  //   cdef float doneRotation = 0

  //   for i in range(1, normals.length):
  //       if angleVec3(tangents.data + i, tangents.data + i - 1) < 0.001:
  //           rotateAroundAxisVec3(normals.data + i, &normal, tangents.data + i, doneRotation)
  //       else:
  //           angle = remainingRotation / (normals.length - i)
  //           normal = normals.data[i]
  //           rotateAroundAxisVec3(normals.data + i, &normal, tangents.data + i, angle +
  //               doneRotation)
  //           remainingRotation -= angle
  //           doneRotation += angle

  float remaining_rotation = full_angle;
  float done_rotation = 0.0f;
  for (const int i : IndexRange(1, normals.size() - 1)) {
    if (angle_v3v3(tangents[i], tangents[i - 1]) < 0.001f) {
      normals[i] = rotate_around_axis(normals[i], tangents[i], done_rotation);
    }
    else {
      const float angle = remaining_rotation / (normals.size() - i);
      normals[i] = rotate_around_axis(normals[i], tangents[i], angle + done_rotation);
      remaining_rotation -= angle;
      done_rotation += angle;
    }
  }
}

static void make_normals_cyclic(Span<float3> tangents, MutableSpan<float3> normals)
{
  const float3 last_normal = propagate_normal(normals.last(), tangents.last(), tangents.first());

  float angle = angle_normalized_v3v3(normals.first(), last_normal);

  const float3 cross = float3::cross(normals.first(), last_normal);
  if (float3::dot(cross, tangents.first()) <= 0.0f) {
    angle = -angle;
  }

  apply_rotation_gradient(tangents, normals, -angle);
}

/* This algorithm is a copy from animation nodes bezier normal calculation.
 * TODO: Explore different methods. */
static void evaluate_normals(Span<float3> tangents,
                             const bool is_cyclic,
                             MutableSpan<float3> normals)
{
  if (normals.size() == 1) {
    normals.first() = float3(1.0f, 0.0f, 0.0f);
    return;
  }

  /* Start by calculating a simple normal for the first point. */
  normals[0] = initial_normal(tangents[0]);

  /* Then propogate that normal along the spline. */
  for (const int i : IndexRange(1, normals.size() - 1)) {
    normals[i] = propagate_normal(normals[i - 1], tangents[i - 1], tangents[i]);
  }

  for (const float3 &normal : normals) {
    BLI_ASSERT_UNIT_V3(normal);
  }

  if (is_cyclic) {
    make_normals_cyclic(tangents, normals);
  }

  for (const float3 &normal : normals) {
    BLI_ASSERT_UNIT_V3(normal);
  }
}

void BezierSpline::ensure_evaluation_cache() const
{
  /* TODO: Consider separating a tangent dirty tag from the position and tangent cache. */
  if (!this->cache_dirty_) {
    return;
  }

  std::lock_guard<std::mutex> lock(this->cache_mutex_);
  if (!this->cache_dirty_) {
    return;
  }

  const int points_len = this->evaluated_points_size();
  this->evaluated_positions_cache_.resize(points_len);
  this->evaluated_tangents_cache_.resize(points_len);
  this->evaluated_normals_cache_.resize(points_len);

  evaluate_positions(
      this->control_points, this->resolution_u, this->is_cyclic, this->evaluated_positions_cache_);

  calculate_tangents(this->control_points,
                     this->evaluated_positions_cache_,
                     this->is_cyclic,
                     this->evaluated_tangents_cache_);

  evaluate_normals(
      this->evaluated_tangents_cache_, this->is_cyclic, this->evaluated_normals_cache_);

  this->cache_dirty_ = false;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name NURBS Spline
 * \{ */

/** \} */

/* -------------------------------------------------------------------- */
/** \name Poly Spline
 * \{ */

/** \} */
