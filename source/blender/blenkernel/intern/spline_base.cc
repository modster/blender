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

#include "BKE_spline.hh"

using blender::Array;
using blender::float3;
using blender::float4x4;
using blender::IndexRange;
using blender::MutableSpan;
using blender::Span;
using blender::Vector;

Spline::Type Spline::type() const
{
  return this->type_;
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

int Spline::segments_size() const
{
  const int points_len = this->size();

  return this->is_cyclic ? points_len : points_len - 1;
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
  const int next_index = (index == this->size() - 1) ? 0 : index + 1;

  const float previous_length = (index == 0) ? 0.0f : lengths[index - 1];
  const float factor = (length - previous_length) / (lengths[index] - previous_length);

  return LookupResult{index, next_index, factor};
}
