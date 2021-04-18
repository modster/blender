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
using blender::float4x4;
using blender::IndexRange;
using blender::MutableSpan;
using blender::Span;
using blender::Vector;

/* -------------------------------------------------------------------- */
/** \name Utilities
 * \{ */

/** \} */

/* -------------------------------------------------------------------- */
/** \name General Curve Functions
 * \{ */

DCurve *DCurve::copy()
{
  DCurve *new_curve = new DCurve();

  for (SplinePtr &spline : this->splines) {
    new_curve->splines.append(spline->copy());
  }

  return new_curve;
}

void DCurve::translate(const float3 translation)
{
  for (SplinePtr &spline : this->splines) {
    if (BezierSpline *bezier_spline = dynamic_cast<BezierSpline *>(spline.get())) {
      for (float3 &position : bezier_spline->positions()) {
        position += translation;
      }
      for (float3 &handle_position : bezier_spline->handle_positions_start()) {
        handle_position += translation;
      }
      for (float3 &handle_position : bezier_spline->handle_positions_end()) {
        handle_position += translation;
      }
    }
    else if (PolySpline *poly_spline = dynamic_cast<PolySpline *>(spline.get())) {
      for (float3 &position : poly_spline->positions()) {
        position += translation;
      }
    }
    spline->mark_cache_invalid();
  }
}

void DCurve::transform(const float4x4 &matrix)
{
  for (SplinePtr &spline : this->splines) {
    if (BezierSpline *bezier_spline = dynamic_cast<BezierSpline *>(spline.get())) {
      for (float3 &position : bezier_spline->positions()) {
        position = matrix * position;
      }
      for (float3 &handle_position : bezier_spline->handle_positions_start()) {
        handle_position = matrix * handle_position;
      }
      for (float3 &handle_position : bezier_spline->handle_positions_end()) {
        handle_position = matrix * handle_position;
      }
    }
    else if (PolySpline *poly_spline = dynamic_cast<PolySpline *>(spline.get())) {
      for (float3 &position : poly_spline->positions()) {
        position = matrix * position;
      }
    }
    spline->mark_cache_invalid();
  }
}

static BezierSpline::HandleType handle_type_from_dna_bezt(const eBezTriple_Handle dna_handle_type)
{
  switch (dna_handle_type) {
    case HD_FREE:
      return BezierSpline::Free;
    case HD_AUTO:
      return BezierSpline::Auto;
    case HD_VECT:
      return BezierSpline::Vector;
    case HD_ALIGN:
      return BezierSpline::Align;
    case HD_AUTO_ANIM:
      return BezierSpline::Auto;
    case HD_ALIGN_DOUBLESIDE:
      return BezierSpline::Align;
  }
  BLI_assert_unreachable();
  return BezierSpline::Auto;
}

static Spline::NormalCalculationMode normal_mode_from_dna_curve(const int twist_mode)
{
  switch (twist_mode) {
    case CU_TWIST_Z_UP:
      return Spline::NormalCalculationMode::ZUp;
    case CU_TWIST_MINIMUM:
      return Spline::NormalCalculationMode::Minimum;
    case CU_TWIST_TANGENT:
      return Spline::NormalCalculationMode::Tangent;
  }
  BLI_assert_unreachable();
  return Spline::NormalCalculationMode::Minimum;
}

DCurve *dcurve_from_dna_curve(const Curve &dna_curve)
{
  DCurve *curve = new DCurve();

  const ListBase *nurbs = BKE_curve_nurbs_get(&const_cast<Curve &>(dna_curve));

  curve->splines.reserve(BLI_listbase_count(nurbs));

  LISTBASE_FOREACH (const Nurb *, nurb, nurbs) {
    switch (nurb->type) {
      case CU_BEZIER: {
        std::unique_ptr<BezierSpline> spline = std::make_unique<BezierSpline>();
        spline->set_resolution(nurb->resolu);
        spline->type = Spline::Type::Bezier;
        spline->is_cyclic = nurb->flagu & CU_NURB_CYCLIC;

        /* TODO: Optimize by reserving the correct size. */
        for (const BezTriple &bezt : Span(nurb->bezt, nurb->pntsu)) {
          spline->add_point(bezt.vec[1],
                            handle_type_from_dna_bezt((eBezTriple_Handle)bezt.h1),
                            bezt.vec[0],
                            handle_type_from_dna_bezt((eBezTriple_Handle)bezt.h2),
                            bezt.vec[2],
                            bezt.radius,
                            bezt.tilt);
        }

        curve->splines.append(std::move(spline));
        break;
      }
      case CU_NURBS: {
        break;
      }
      case CU_POLY: {
        std::unique_ptr<PolySpline> spline = std::make_unique<PolySpline>();
        spline->type = Spline::Type::Poly;
        spline->is_cyclic = nurb->flagu & CU_NURB_CYCLIC;

        for (const BPoint &bp : Span(nurb->bp, nurb->pntsu)) {
          spline->add_point(bp.vec, bp.radius, bp.tilt);
        }

        curve->splines.append(std::move(spline));
        break;
      }
      default: {
        BLI_assert_unreachable();
        break;
      }
    }
  }

  /* TODO: Decide whether to store this in the spline or the curve. */
  const Spline::NormalCalculationMode normal_mode = normal_mode_from_dna_curve(
      dna_curve.twist_mode);
  for (SplinePtr &spline : curve->splines) {
    spline->normal_mode = normal_mode;
  }

  return curve;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Spline
 * \{ */

/**
 * Mark all caches for recomputation. This must be called after any operation that would
 * change the generated positions, tangents, normals, mapping, etc. of the evaluated points.
 */
void Spline::mark_cache_invalid()
{
  base_cache_dirty_ = true;
  tangent_cache_dirty_ = true;
  normal_cache_dirty_ = true;
  length_cache_dirty_ = true;
}

int Spline::evaluated_edges_size() const
{
  const int points_len = this->evaluated_points_size();

  return this->is_cyclic ? points_len : points_len - 1;
}

float Spline::length() const
{
  return this->evaluated_lengths().last();
}

Span<float3> Spline::evaluated_positions() const
{
  this->ensure_base_cache();
  return evaluated_positions_cache_;
}

/**
 * Returns non-owning access to the cache of mappings from the evaluated points to
 * the corresponing control points. Unless the spline is cyclic, the last control point
 * index will never be included as an index.
 */
Span<PointMapping> Spline::evaluated_mappings() const
{
  this->ensure_base_cache();
#ifdef DEBUG
  if (evaluated_mapping_cache_.last().control_point_index == this->size() - 1) {
    BLI_assert(this->is_cyclic);
  }
#endif
  return evaluated_mapping_cache_;
}

static void accumulate_lengths(Span<float3> positions,
                               const bool is_cyclic,
                               MutableSpan<float> lengths)
{
  float length = 0.0f;
  for (const int i : IndexRange(positions.size() - 1)) {
    length += float3::distance(positions[i], positions[i + 1]);
    lengths[i] = length;
  }
  if (is_cyclic) {
    lengths.last() = length + float3::distance(positions.last(), positions.first());
  }
}

/**
 * Return non-owning access to the cache of accumulated lengths along the spline. Each item is the
 * length of the subsequent segment, i.e. the first value is the length of the first segment rather
 * than 0. This calculation is rather trivial, and only depends on the evaluated positions.
 * However, the results are used often, so it makes sense to cache it.
 */
Span<float> Spline::evaluated_lengths() const
{
  if (!this->length_cache_dirty_) {
    return evaluated_lengths_cache_;
  }

  std::lock_guard lock{this->length_cache_mutex_};
  if (!this->length_cache_dirty_) {
    return evaluated_lengths_cache_;
  }

  const int total = this->evaluated_edges_size();
  this->evaluated_lengths_cache_.resize(total);

  Span<float3> positions = this->evaluated_positions();
  accumulate_lengths(positions, this->is_cyclic, this->evaluated_lengths_cache_);

  this->length_cache_dirty_ = false;
  return evaluated_lengths_cache_;
}

/* TODO: Optimize this along with the function below. */
static float3 direction_bisect(const float3 &prev, const float3 &middle, const float3 &next)
{
  const float3 dir_prev = (middle - prev).normalized();
  const float3 dir_next = (next - middle).normalized();

  return (dir_prev + dir_next).normalized();
}

static void calculate_tangents(Span<float3> positions,
                               const bool is_cyclic,
                               MutableSpan<float3> tangents)
{
  if (positions.size() == 1) {
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
    tangents.first() = (positions[1] - positions[0]).normalized();
    tangents.last() = (positions.last() - positions[positions.size() - 1]).normalized();
  }
}

/**
 * Return non-owning access to the direction of the curve at each evaluated point.
 */
Span<float3> Spline::evaluated_tangents() const
{
  if (!this->tangent_cache_dirty_) {
    return evaluated_tangents_cache_;
  }

  std::lock_guard lock{this->tangent_cache_mutex_};
  if (!this->tangent_cache_dirty_) {
    return evaluated_tangents_cache_;
  }

  const int total = this->evaluated_points_size();
  this->evaluated_tangents_cache_.resize(total);

  Span<float3> positions = this->evaluated_positions();

  calculate_tangents(positions, this->is_cyclic, this->evaluated_tangents_cache_);

  this->correct_end_tangents();

  this->tangent_cache_dirty_ = false;
  return evaluated_tangents_cache_;
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

  const float3 new_normal = rotate_around_axis(last_normal, axis, angle);

  return project_on_center_plane(new_normal, current_tangent).normalized();
}

static void apply_rotation_gradient(Span<float3> tangents,
                                    MutableSpan<float3> normals,
                                    const float full_angle)
{

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
 * TODO: Explore different methods, this also doesn't work right now. */
static void calculate_normals_minimum_twist(Span<float3> tangents,
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

  if (is_cyclic) {
    make_normals_cyclic(tangents, normals);
  }
}

static void calculate_normals_z_up(Span<float3> tangents, MutableSpan<float3> normals)
{
  for (const int i : normals.index_range()) {
    normals[i] = float3::cross(tangents[i], float3(0.0f, 0.0f, 1.0f)).normalized();
  }
}

/**
 * Return non-owning access to the direction vectors perpendicular to the tangents at every
 * evaluated point. The method used to generate the normal vectors depends on Spline.normal_mode.
 */
Span<float3> Spline::evaluated_normals() const
{
  if (!this->normal_cache_dirty_) {
    return evaluated_normals_cache_;
  }

  std::lock_guard lock{this->normal_cache_mutex_};
  if (!this->normal_cache_dirty_) {
    return evaluated_normals_cache_;
  }

  const int total = this->evaluated_points_size();
  this->evaluated_normals_cache_.resize(total);

  Span<float3> tangents = this->evaluated_tangents();
  switch (this->normal_mode) {
    case NormalCalculationMode::Minimum:
      calculate_normals_minimum_twist(tangents, is_cyclic, this->evaluated_normals_cache_);
      break;
    case NormalCalculationMode::ZUp:
      calculate_normals_z_up(tangents, this->evaluated_normals_cache_);
      break;
    case NormalCalculationMode::Tangent:
      // calculate_normals_tangent(tangents, this->evaluated_normals_cache_);
      break;
  }

  this->normal_cache_dirty_ = false;
  return evaluated_normals_cache_;
}

Spline::LookupResult Spline::lookup_evaluated_factor(const float factor) const
{
  return this->lookup_evaluated_length(this->length() * factor);
}

/* TODO: Support extrapolation somehow. */
Spline::LookupResult Spline::lookup_evaluated_length(const float length) const
{
  BLI_assert(length >= 0.0f && length <= this->length());

  Span<float> lengths = this->evaluated_lengths();

  const float *offset = std::lower_bound(lengths.begin(), lengths.end(), length);
  const int index = offset - lengths.begin();

  const float segment_length = lengths[index];
  const float previous_length = (index == 0) ? 0.0f : lengths[index - 1];
  const float factor = (length - previous_length) / (segment_length - previous_length);

  return LookupResult{index, factor};
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Bezier Spline
 * \{ */

SplinePtr BezierSpline::copy() const
{
  SplinePtr new_spline = std::make_unique<BezierSpline>(*this);

  return new_spline;
}

int BezierSpline::size() const
{
  const int size = this->positions_.size();
  BLI_assert(this->handle_types_start_.size() == size);
  BLI_assert(this->handle_positions_start_.size() == size);
  BLI_assert(this->handle_types_end_.size() == size);
  BLI_assert(this->handle_positions_end_.size() == size);
  BLI_assert(this->radii_.size() == size);
  BLI_assert(this->tilts_.size() == size);
  return size;
}

int BezierSpline::resolution() const
{
  return this->resolution_u_;
}

void BezierSpline::set_resolution(const int value)
{
  this->resolution_u_ = value;
  this->mark_cache_invalid();
}

MutableSpan<float3> BezierSpline::positions()
{
  return this->positions_;
}
Span<float3> BezierSpline::positions() const
{
  return this->positions_;
}
MutableSpan<float> BezierSpline::radii()
{
  return this->radii_;
}
Span<float> BezierSpline::radii() const
{
  return this->radii_;
}
MutableSpan<float> BezierSpline::tilts()
{
  return this->tilts_;
}
Span<float> BezierSpline::tilts() const
{
  return this->tilts_;
}
Span<BezierSpline::HandleType> BezierSpline::handle_types_start() const
{
  return this->handle_types_start_;
}
MutableSpan<BezierSpline::HandleType> BezierSpline::handle_types_start()
{
  return this->handle_types_start_;
}
Span<float3> BezierSpline::handle_positions_start() const
{
  return this->handle_positions_start_;
}
MutableSpan<float3> BezierSpline::handle_positions_start()
{
  return this->handle_positions_start_;
}
Span<BezierSpline::HandleType> BezierSpline::handle_types_end() const
{
  return this->handle_types_end_;
}
MutableSpan<BezierSpline::HandleType> BezierSpline::handle_types_end()
{
  return this->handle_types_end_;
}
Span<float3> BezierSpline::handle_positions_end() const
{
  return this->handle_positions_end_;
}
MutableSpan<float3> BezierSpline::handle_positions_end()
{
  return this->handle_positions_end_;
}

void BezierSpline::add_point(const float3 position,
                             const HandleType handle_type_start,
                             const float3 handle_position_start,
                             const HandleType handle_type_end,
                             const float3 handle_position_end,
                             const float radius,
                             const float tilt)
{
  handle_types_start_.append(handle_type_start);
  handle_positions_start_.append(handle_position_start);
  positions_.append(position);
  handle_types_end_.append(handle_type_end);
  handle_positions_end_.append(handle_position_end);
  radii_.append(radius);
  tilts_.append(tilt);
}

void BezierSpline::drop_front(const int count)
{
  std::cout << __func__ << ": " << count << "\n";
  BLI_assert(this->size() - count > 0);
  this->handle_types_start_.remove(0, count);
  this->handle_positions_start_.remove(0, count);
  this->positions_.remove(0, count);
  this->handle_types_end_.remove(0, count);
  this->handle_positions_end_.remove(0, count);
  this->radii_.remove(0, count);
  this->tilts_.remove(0, count);
  this->mark_cache_invalid();
}

void BezierSpline::drop_back(const int count)
{
  std::cout << __func__ << ": " << count << "\n";
  const int new_size = this->size() - count;
  BLI_assert(new_size > 0);
  this->handle_types_start_.resize(new_size);
  this->handle_positions_start_.resize(new_size);
  this->positions_.resize(new_size);
  this->handle_types_end_.resize(new_size);
  this->handle_positions_end_.resize(new_size);
  this->radii_.resize(new_size);
  this->tilts_.resize(new_size);
  this->mark_cache_invalid();
}

bool BezierSpline::point_is_sharp(const int index) const
{
  return ELEM(handle_types_start_[index], HandleType::Vector, HandleType::Free) ||
         ELEM(handle_types_end_[index], HandleType::Vector, HandleType::Free);
}

bool BezierSpline::handle_start_is_automatic(const int index) const
{
  return ELEM(handle_types_start_[index], HandleType::Free, HandleType::Align);
}

bool BezierSpline::handle_end_is_automatic(const int index) const
{
  return ELEM(handle_types_end_[index], HandleType::Free, HandleType::Align);
}

void BezierSpline::move_control_point(const int index, const blender::float3 new_position)
{
  const float3 position_delta = new_position - positions_[index];
  if (!this->handle_start_is_automatic(index)) {
    handle_positions_start_[index] += position_delta;
  }
  if (!this->handle_end_is_automatic(index)) {
    handle_positions_end_[index] += position_delta;
  }
  positions_[index] = new_position;
}

bool BezierSpline::segment_is_vector(const int index) const
{
  if (index == this->size() - 1) {
    BLI_assert(this->is_cyclic);
    return this->handle_types_end_.last() == HandleType::Vector &&
           this->handle_types_start_.first() == HandleType::Vector;
  }
  return this->handle_types_end_[index] == HandleType::Vector &&
         this->handle_types_start_[index + 1] == HandleType::Vector;
}

int BezierSpline::evaluated_points_size() const
{
  BLI_assert(this->size() > 0);
#ifndef DEBUG
  if (!this->base_cache_dirty_) {
    /* In a non-debug build, assume that the cache's size has not changed, and that any operation
     * that would cause the cache to change its length would also mark the cache dirty. This is
     * checked at the end of this function in a debug build. */
    return this->evaluated_positions_cache_.size();
  }
#endif

  int total_len = 0;
  for (const int i : IndexRange(this->size() - 1)) {
    if (this->segment_is_vector(i)) {
      total_len += 1;
    }
    else {
      total_len += this->resolution_u_;
    }
  }

  if (this->is_cyclic) {
    if (segment_is_vector(this->size() - 1)) {
      total_len++;
    }
    else {
      total_len += this->resolution_u_;
    }
  }
  else {
    /* Since evaulating the bezier doesn't add the final point's position,
     * it must be added manually in the non-cyclic case. */
    total_len++;
  }

  /* Assert that the cache has the correct length in debug mode. */
  if (!this->base_cache_dirty_) {
    BLI_assert(this->evaluated_positions_cache_.size() == total_len);
  }

  return total_len;
}

/**
 * If the spline is not cyclic, the direction for the first and last points is just the
 * direction formed by the corresponding handles and control points. In the unlikely situation
 * that the handles define a zero direction, fallback to using the direction defined by the
 * first and last evaluated segments already calculated in #Spline::evaluated_tangents().
 */
void BezierSpline::correct_end_tangents() const
{
  MutableSpan<float3> tangents(this->evaluated_tangents_cache_);

  if (handle_positions_start_.first() != positions_.first()) {
    tangents.first() = (positions_.first() - handle_positions_start_.first()).normalized();
  }
  if (handle_positions_end_.last() != positions_.last()) {
    tangents.last() = (handle_positions_end_.last() - positions_.last()).normalized();
  }
}

static void bezier_forward_difference_3d(const float3 &point_0,
                                         const float3 &point_1,
                                         const float3 &point_2,
                                         const float3 &point_3,
                                         MutableSpan<float3> result)
{
  const float len = static_cast<float>(result.size());
  const float len_squared = len * len;
  const float len_cubed = len_squared * len;
  BLI_assert(len > 0.0f);

  const float3 rt1 = 3.0f * (point_1 - point_0) / len;
  const float3 rt2 = 3.0f * (point_0 - 2.0f * point_1 + point_2) / len_squared;
  const float3 rt3 = (point_3 - point_0 + 3.0f * (point_1 - point_2)) / len_cubed;

  float3 q0 = point_0;
  float3 q1 = rt1 + rt2 + rt3;
  float3 q2 = 2.0f * rt2 + 6.0f * rt3;
  float3 q3 = 6.0f * rt3;
  for (const int i : result.index_range()) {
    result[i] = q0;
    q0 += q1;
    q1 += q2;
    q2 += q3;
  }
}

static void evaluate_segment_mapping(Span<float3> evaluated_positions,
                                     MutableSpan<PointMapping> mappings,
                                     const int index)
{
  float length = 0.0f;
  mappings[0] = PointMapping{index, 0.0f};
  for (const int i : IndexRange(1, mappings.size() - 1)) {
    length += float3::distance(evaluated_positions[i - 1], evaluated_positions[i]);
    mappings[i] = PointMapping{index, length};
  }

  /* To get the factors instead of the accumulated lengths, divide the mapping factors by the
   * accumulated length. */
  if (length != 0.0f) {
    for (PointMapping &mapping : mappings) {
      mapping.factor /= length;
    }
  }
}

void BezierSpline::evaluate_bezier_segment(const int index,
                                           const int next_index,
                                           int &offset,
                                           MutableSpan<float3> positions,
                                           MutableSpan<PointMapping> mappings) const
{
  if (this->segment_is_vector(index)) {
    positions[offset] = positions_[index];
    mappings[offset] = PointMapping{index, 0.0f};
    offset++;
  }
  else {
    bezier_forward_difference_3d(this->positions_[index],
                                 this->handle_positions_end_[index],
                                 this->handle_positions_start_[next_index],
                                 this->positions_[next_index],
                                 positions.slice(offset, this->resolution_u_));
    evaluate_segment_mapping(positions.slice(offset, this->resolution_u_),
                             mappings.slice(offset, this->resolution_u_),
                             index);
    offset += this->resolution_u_;
  }
}

void BezierSpline::evaluate_bezier_position_and_mapping(MutableSpan<float3> positions,
                                                        MutableSpan<PointMapping> mappings) const
{
  /* TODO: It would also be possible to store an array of offsets to facilitate parallelism here,
   * maybe it is worth it? */
  int offset = 0;
  for (const int i : IndexRange(this->size() - 1)) {
    this->evaluate_bezier_segment(i, i + 1, offset, positions, mappings);
  }

  const int i_last = this->size() - 1;
  if (this->is_cyclic) {
    this->evaluate_bezier_segment(i_last, 0, offset, positions, mappings);
  }
  else {
    /* Since evaulating the bezier doesn't add the final point's position,
     * it must be added manually in the non-cyclic case. */
    positions[offset] = this->positions_.last();
    mappings[offset] = PointMapping{i_last - 1, 1.0f};
    offset++;
  }

  BLI_assert(offset == positions.size());
}

void BezierSpline::ensure_base_cache() const
{
  if (!this->base_cache_dirty_) {
    return;
  }

  std::lock_guard lock{this->base_cache_mutex_};
  if (!this->base_cache_dirty_) {
    return;
  }

  const int total = this->evaluated_points_size();
  this->evaluated_positions_cache_.resize(total);
  this->evaluated_mapping_cache_.resize(total);

  this->evaluate_bezier_position_and_mapping(this->evaluated_positions_cache_,
                                             this->evaluated_mapping_cache_);

  this->base_cache_dirty_ = false;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name NURBS Spline
 * \{ */

SplinePtr NURBSpline::copy() const
{
  SplinePtr new_spline = std::make_unique<NURBSpline>(*this);

  return new_spline;
}

int NURBSpline::size() const
{
  const int size = this->positions_.size();
  BLI_assert(this->radii_.size() == size);
  BLI_assert(this->tilts_.size() == size);
  BLI_assert(this->weights_.size() == size);
  return size;
}

int NURBSpline::resolution() const
{
  return this->resolution_u_;
}

void NURBSpline::set_resolution(const int value)
{
  this->resolution_u_ = value;
  this->mark_cache_invalid();
}

void NURBSpline::add_point(const float3 position,
                           const float radius,
                           const float tilt,
                           const float weight)
{
  this->positions_.append(position);
  this->radii_.append(radius);
  this->tilts_.append(tilt);
  this->weights_.append(weight);
}

void NURBSpline::drop_front(const int count)
{
  BLI_assert(this->size() - count > 0);
  this->positions_.remove(0, count);
  this->radii_.remove(0, count);
  this->tilts_.remove(0, count);
  this->weights_.remove(0, count);
  this->mark_cache_invalid();
}

void NURBSpline::drop_back(const int count)
{
  const int new_size = this->size() - count;
  BLI_assert(new_size > 0);
  this->positions_.resize(new_size);
  this->radii_.resize(new_size);
  this->tilts_.resize(new_size);
  this->weights_.resize(new_size);
  this->mark_cache_invalid();
}

MutableSpan<float3> NURBSpline::positions()
{
  return this->positions_;
}
Span<float3> NURBSpline::positions() const
{
  return this->positions_;
}
MutableSpan<float> NURBSpline::radii()
{
  return this->radii_;
}
Span<float> NURBSpline::radii() const
{
  return this->radii_;
}
MutableSpan<float> NURBSpline::tilts()
{
  return this->tilts_;
}
Span<float> NURBSpline::tilts() const
{
  return this->tilts_;
}
MutableSpan<float> NURBSpline::weights()
{
  return this->weights_;
}
Span<float> NURBSpline::weights() const
{
  return this->weights_;
}

int NURBSpline::evaluated_points_size() const
{
  return 0;
}

void NURBSpline::correct_end_tangents() const
{
}

void NURBSpline::ensure_base_cache() const
{
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Poly Spline
 * \{ */

SplinePtr PolySpline::copy() const
{
  SplinePtr new_spline = std::make_unique<PolySpline>(*this);

  return new_spline;
}

int PolySpline::size() const
{
  const int size = this->positions_.size();
  BLI_assert(this->radii_.size() == size);
  BLI_assert(this->tilts_.size() == size);
  return size;
}

int PolySpline::resolution() const
{
  return 1;
}

void PolySpline::set_resolution(const int UNUSED(value))
{
  /* Poly curve has no resolution, there is just one evaluated point per control point. */
}

void PolySpline::add_point(const float3 position, const float radius, const float tilt)
{
  this->positions_.append(position);
  this->radii_.append(radius);
  this->tilts_.append(tilt);
}

void PolySpline::drop_front(const int count)
{
  BLI_assert(this->size() - count > 0);
  this->positions_.remove(0, count);
  this->radii_.remove(0, count);
  this->tilts_.remove(0, count);
  this->mark_cache_invalid();
}

void PolySpline::drop_back(const int count)
{
  const int new_size = this->size() - count;
  BLI_assert(new_size > 0);
  this->positions_.resize(new_size);
  this->radii_.resize(new_size);
  this->tilts_.resize(new_size);
  this->mark_cache_invalid();
}

MutableSpan<float3> PolySpline::positions()
{
  return this->positions_;
}
Span<float3> PolySpline::positions() const
{
  return this->positions_;
}
MutableSpan<float> PolySpline::radii()
{
  return this->radii_;
}
Span<float> PolySpline::radii() const
{
  return this->radii_;
}
MutableSpan<float> PolySpline::tilts()
{
  return this->tilts_;
}
Span<float> PolySpline::tilts() const
{
  return this->tilts_;
}

int PolySpline::evaluated_points_size() const
{
  return this->size();
}

void PolySpline::correct_end_tangents() const
{
}

/* TODO: Consider refactoring to avoid copying and "mapping" for poly splines. */
void PolySpline::ensure_base_cache() const
{
  if (!this->base_cache_dirty_) {
    return;
  }

  std::lock_guard lock{this->base_cache_mutex_};
  if (!this->base_cache_dirty_) {
    return;
  }

  const int total = this->evaluated_points_size();
  this->evaluated_positions_cache_.resize(total);
  this->evaluated_mapping_cache_.resize(total);

  MutableSpan<float3> positions = this->evaluated_positions_cache_.as_mutable_span();
  MutableSpan<PointMapping> mappings = this->evaluated_mapping_cache_.as_mutable_span();

  for (const int i : positions.index_range()) {
    positions[i] = this->positions_[i];
    mappings[i].control_point_index = i;
    mappings[i].factor = 0.0f;
  }

  this->base_cache_dirty_ = false;
}

/** \} */
