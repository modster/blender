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

#include "BKE_spline.hh"

using blender::Array;
using blender::float3;
using blender::float4x4;
using blender::IndexRange;
using blender::MutableSpan;
using blender::Span;
using blender::Vector;

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

void BezierSpline::mark_cache_invalid()
{
  this->base_cache_dirty_ = true;
  this->tangent_cache_dirty_ = true;
  this->normal_cache_dirty_ = true;
  this->length_cache_dirty_ = true;
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
                                     MutableSpan<float> mappings,
                                     const int index)
{
  float length = 0.0f;
  mappings[0] = index;
  for (const int i : IndexRange(1, mappings.size() - 1)) {
    length += float3::distance(evaluated_positions[i - 1], evaluated_positions[i]);
    mappings[i] = length;
  }

  /* To get the factors instead of the accumulated lengths, divide the mapping factors by the
   * accumulated length. */
  if (length != 0.0f) {
    for (float &mapping : mappings) {
      mapping = mapping / length + index;
    }
  }
}

void BezierSpline::evaluate_bezier_segment(const int index,
                                           const int next_index,
                                           int &offset,
                                           MutableSpan<float3> positions,
                                           MutableSpan<float> mappings) const
{
  if (this->segment_is_vector(index)) {
    positions[offset] = positions_[index];
    mappings[offset] = index;
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

void BezierSpline::evaluate_bezier_position_and_mapping() const
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
  this->evaluated_mappings_cache_.resize(total);

  MutableSpan<float3> positions = this->evaluated_positions_cache_;
  MutableSpan<float> mappings = this->evaluated_mappings_cache_;

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
    mappings[offset] = i_last;
    offset++;
  }

  if (this->is_cyclic) {
    if (mappings.last() >= this->size()) {
      mappings.last() = 0.0f;
    }
  }

  BLI_assert(offset == positions.size());

  this->base_cache_dirty_ = false;
}

/**
 * Returns non-owning access to an array of values ontains the information necessary to
 * interpolate values from the original control points to evaluated points. The control point
 * index is the integer part of each value, and the factor used for interpolating to the next
 * control point is the remaining factional part.
 */
Span<float> BezierSpline::evaluated_mappings() const
{
  this->evaluate_bezier_position_and_mapping();
  return this->evaluated_mappings_cache_;
}

Span<float3> BezierSpline::evaluated_positions() const
{
  this->evaluate_bezier_position_and_mapping();
  return this->evaluated_positions_cache_;
}

BezierSpline::InterpolationData BezierSpline::interpolation_data_from_map(const float map) const
{
  const int points_len = this->size();
  const int index = std::floor(map);
  if (index == points_len) {
    BLI_assert(this->is_cyclic);
    return InterpolationData{points_len - 1, 0, 1.0f};
  }
  if (index == points_len - 1) {
    return InterpolationData{points_len - 2, points_len - 1, 1.0f};
  }
  return InterpolationData{index, index + 1, map - index};
}

template<typename T>
static void interpolate_to_evaluated_points_impl(Span<float> mappings,
                                                 const blender::VArray<T> &source_data,
                                                 MutableSpan<T> result_data)
{
  const int points_len = source_data.size();
  blender::attribute_math::DefaultMixer<T> mixer(result_data);

  for (const int i : result_data.index_range()) {
    const int index = std::floor(mappings[i]);
    const int next_index = (index == points_len - 1) ? 0 : index + 1;
    const float factor = mappings[i] - index;

    const T &value = source_data[index];
    const T &next_value = source_data[next_index];

    mixer.mix_in(i, value, 1.0f - factor);
    mixer.mix_in(i, next_value, factor);
  }

  mixer.finalize();
}

blender::fn::GVArrayPtr BezierSpline::interpolate_to_evaluated_points(
    const blender::fn::GVArray &source_data) const
{
  BLI_assert(source_data.size() == this->size());
  Span<float> mappings = this->evaluated_mappings();

  blender::fn::GVArrayPtr new_varray;
  blender::attribute_math::convert_to_static_type(source_data.type(), [&](auto dummy) {
    using T = decltype(dummy);
    if constexpr (!std::is_void_v<blender::attribute_math::DefaultMixer<T>>) {
      Array<T> values(this->evaluated_points_size());
      interpolate_to_evaluated_points_impl<T>(mappings, source_data.typed<T>(), values);
      new_varray = std::make_unique<blender::fn::GVArray_For_ArrayContainer<Array<T>>>(
          std::move(values));
    }
  });

  return new_varray;
}
