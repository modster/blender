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
#include "BLI_span.hh"
#include "BLI_task.hh"

#include "BKE_spline.hh"

using blender::Array;
using blender::float3;
using blender::IndexRange;
using blender::MutableSpan;
using blender::Span;

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
  return this->resolution_;
}

void BezierSpline::set_resolution(const int value)
{
  this->resolution_ = value;
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

void BezierSpline::move_control_point(const int index, const float3 new_position)
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
  this->offset_cache_dirty_ = true;
  this->position_cache_dirty_ = true;
  this->mapping_cache_dirty_ = true;
  this->tangent_cache_dirty_ = true;
  this->normal_cache_dirty_ = true;
  this->length_cache_dirty_ = true;
}

int BezierSpline::evaluated_points_size() const
{
  const int points_len = this->size();
  BLI_assert(points_len > 0);

  const int last_offset = this->control_point_offsets().last();
  if (this->is_cyclic) {
    return last_offset + (this->segment_is_vector(points_len - 1) ? 0 : this->resolution_);
  }

  return last_offset + 1;
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

void BezierSpline::evaluate_bezier_segment(const int index,
                                           const int next_index,
                                           MutableSpan<float3> positions) const
{
  if (this->segment_is_vector(index)) {
    positions.first() = this->positions_[index];
  }
  else {
    bezier_forward_difference_3d(this->positions_[index],
                                 this->handle_positions_end_[index],
                                 this->handle_positions_start_[next_index],
                                 this->positions_[next_index],
                                 positions);
  }
}

/**
 * Returns access to a cache of offsets into the evaluated point array for each control point.
 * This is important because while most control point edges generate the number of edges specified
 * by the resolution, vector segments only generate one edge.
 */
Span<int> BezierSpline::control_point_offsets() const
{
  if (!this->offset_cache_dirty_) {
    return this->offset_cache_;
  }

  std::lock_guard lock{this->offset_cache_mutex_};
  if (!this->offset_cache_dirty_) {
    return this->offset_cache_;
  }

  const int points_len = this->size();
  this->offset_cache_.resize(points_len);

  MutableSpan<int> offsets = this->offset_cache_;

  int offset = 0;
  for (const int i : IndexRange(points_len - 1)) {
    offsets[i] = offset;
    offset += this->segment_is_vector(i) ? 1 : this->resolution_;
  }
  offsets.last() = offset;

  this->offset_cache_dirty_ = false;
  return offsets;
}

/**
 * Returns non-owning access to an array of values ontains the information necessary to
 * interpolate values from the original control points to evaluated points. The control point
 * index is the integer part of each value, and the factor used for interpolating to the next
 * control point is the remaining factional part.
 */
Span<float> BezierSpline::evaluated_mappings() const
{
  if (!this->mapping_cache_dirty_) {
    return this->evaluated_mapping_cache_;
  }

  std::lock_guard lock{this->mapping_cache_mutex_};
  if (!this->mapping_cache_dirty_) {
    return this->evaluated_mapping_cache_;
  }

  const int size = this->size();
  const int eval_size = this->evaluated_points_size();
  this->evaluated_mapping_cache_.resize(eval_size);
  MutableSpan<float> mappings = this->evaluated_mapping_cache_;

  Span<int> offsets = this->control_point_offsets();
  Span<float> lengths = this->evaluated_lengths();

  /* Subtract one from the index into the lengths array to get the length
   * at the start point rather than the length at the end of the edge. */

  const float first_segment_len = lengths[offsets[1] - 1];
  for (const int eval_index : IndexRange(0, offsets[1])) {
    const float point_len = eval_index == 0 ? 0.0f : lengths[eval_index - 1];
    const float length_factor = (first_segment_len == 0.0f) ? 0.0f : 1.0f / first_segment_len;

    mappings[eval_index] = point_len * length_factor;
  }

  const int grain_size = std::max(512 / this->resolution_, 1);
  blender::parallel_for(IndexRange(1, size - 2), grain_size, [&](IndexRange range) {
    for (const int i : range) {
      const float segment_start_len = lengths[offsets[i] - 1];
      const float segment_end_len = lengths[offsets[i + 1] - 1];
      const float segment_len = segment_end_len - segment_start_len;
      const float length_factor = (segment_len == 0.0f) ? 0.0f : 1.0f / segment_len;

      for (const int eval_index : IndexRange(offsets[i], offsets[i + 1] - offsets[i])) {
        const float factor = (lengths[eval_index - 1] - segment_start_len) * length_factor;
        mappings[eval_index] = i + factor;
      }
    }
  });

  if (this->is_cyclic) {
    const float segment_start_len = lengths[offsets.last() - 1];
    const float segment_end_len = this->length();
    const float segment_len = segment_end_len - segment_start_len;
    const float length_factor = (segment_len == 0.0f) ? 0.0f : 1.0f / segment_len;

    for (const int eval_index : IndexRange(offsets.last(), eval_size - offsets.last())) {
      const float factor = (lengths[eval_index - 1] - segment_start_len) * length_factor;
      mappings[eval_index] = size - 1 + factor;
    }
    mappings.last() = 0.0f;
  }
  else {
    mappings.last() = size - 1;
  }

  this->mapping_cache_dirty_ = false;
  return this->evaluated_mapping_cache_;
}

Span<float3> BezierSpline::evaluated_positions() const
{
  if (!this->position_cache_dirty_) {
    return this->evaluated_position_cache_;
  }

  std::lock_guard lock{this->position_cache_mutex_};
  if (!this->position_cache_dirty_) {
    return this->evaluated_position_cache_;
  }

  const int eval_total = this->evaluated_points_size();
  this->evaluated_position_cache_.resize(eval_total);

  MutableSpan<float3> positions = this->evaluated_position_cache_;

  Span<int> offsets = this->control_point_offsets();
  BLI_assert(offsets.last() <= eval_total);

  const int grain_size = std::max(512 / this->resolution_, 1);
  blender::parallel_for(IndexRange(this->size() - 1), grain_size, [&](IndexRange range) {
    for (const int i : range) {
      this->evaluate_bezier_segment(
          i, i + 1, positions.slice(offsets[i], offsets[i + 1] - offsets[i]));
    }
  });

  const int i_last = this->size() - 1;
  if (this->is_cyclic) {
    this->evaluate_bezier_segment(i_last, 0, positions.slice(offsets.last(), this->resolution_));
  }
  else {
    /* Since evaulating the bezier segment doesn't add the final point,
     * it must be added manually in the non-cyclic case. */
    positions.last() = this->positions_.last();
  }

  this->position_cache_dirty_ = false;
  return this->evaluated_position_cache_;
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
  /* TODO: Use a set of functions mix2 in attribute_math instead of DefaultMixer. */
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
