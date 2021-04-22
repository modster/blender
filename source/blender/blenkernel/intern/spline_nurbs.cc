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
#include "BLI_virtual_array.hh"

#include "BKE_attribute_math.hh"
#include "BKE_spline.hh"

using blender::Array;
using blender::float3;
using blender::float4x4;
using blender::IndexRange;
using blender::MutableSpan;
using blender::Span;
using blender::Vector;

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

uint8_t NURBSpline::order() const
{
  return this->order_;
}

void NURBSpline::set_order(const uint8_t value)
{
  /* TODO: Check the spline length. */
  BLI_assert(value >= 2 && value <= 6);
  this->order_ = value;
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
  this->knots_dirty_ = true;
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
  return this->resolution_u_ * this->segments_size();
}

void NURBSpline::correct_end_tangents() const
{
}

bool NURBSpline::check_valid_size_and_order() const
{
  if (this->size() < this->order_) {
    return false;
  }

  if (!this->is_cyclic && this->knots_mode == KnotsMode::Bezier) {
    if (this->order_ == 4) {
      if (this->size() < 5) {
        return false;
      }
    }
    else if (this->order_ != 3) {
      return false;
    }
  }

  return true;
}

int NURBSpline::knots_size() const
{
  const int size = this->size() + this->order_;
  return this->is_cyclic ? size + this->order_ - 1 : size;
}

void NURBSpline::calculate_knots() const
{
  const KnotsMode mode = this->knots_mode;
  const int length = this->size();
  const int order = this->order_;

  this->knots_.resize(this->knots_size());

  MutableSpan<float> knots = this->knots_;

  if (mode == NURBSpline::KnotsMode::Normal || this->is_cyclic) {
    for (const int i : knots.index_range()) {
      knots[i] = static_cast<float>(i);
    }
  }
  else if (mode == NURBSpline::KnotsMode::EndPoint) {
    float k = 0.0f;
    for (const int i : IndexRange(1, knots.size())) {
      knots[i - 1] = k;
      if (i >= order && i <= length) {
        k += 1.0f;
      }
    }
  }
  else if (mode == NURBSpline::KnotsMode::Bezier) {
    BLI_assert(ELEM(order, 3, 4));
    if (order == 3) {
      float k = 0.6f;
      for (const int i : knots.index_range()) {
        if (i >= order && i <= length) {
          k += 0.5f;
        }
        knots[i] = std::floor(k);
      }
    }
    else {
      float k = 0.34f;
      for (const int i : knots.index_range()) {
        knots[i] = std::floor(k);
        k += 1.0f / 3.0f;
      }
    }
  }

  if (this->is_cyclic) {
    const int b = length + order - 1;
    if (order > 2) {
      for (const int i : IndexRange(1, order - 2)) {
        if (knots[b] != knots[b - i]) {
          if (i == order - 1) {
            knots[length + order - 2] += 1.0f;
            break;
          }
        }
      }
    }

    int c = order;
    for (int i = b; i < this->knots_size(); i++) {
      knots[i] = knots[i - 1] + (knots[c] - knots[c - 1]);
      c--;
    }
  }
}

Span<float> NURBSpline::knots() const
{
  if (!this->knots_dirty_) {
    BLI_assert(this->knots_.size() == this->size() + this->order_);
    return this->knots_;
  }

  std::lock_guard lock{this->knots_mutex_};
  if (!this->knots_dirty_) {
    BLI_assert(this->knots_.size() == this->size() + this->order_);
    return this->knots_;
  }

  this->calculate_knots();

  this->base_cache_dirty_ = false;

  return this->knots_;
}

/* TODO: Better variables names, simplify logic once it works. */
static void nurb_basis(const float parameter,
                       const int points_len,
                       const int order,
                       Span<float> knots,
                       MutableSpan<float> basis,
                       int &start,
                       int &end)
{
  /* Clamp parameter due to floating point inaccuracy. TODO: Look into using doubles. */
  const float t = std::clamp(parameter, knots[0], knots[points_len + order - 1]);

  int i1 = 0;
  int i2 = 0;
  for (int i = 0; i < points_len + order - 1; i++) {
    if ((knots[i] != knots[i + 1]) && (t >= knots[i]) && (t <= knots[i + 1])) {
      basis[i] = 1.0f;
      i1 = std::max(i - order - 1, 0);
      i2 = i;
      i++;
      while (i < points_len + order - 1) {
        basis[i] = 0.0f;
        i++;
      }
      break;
    }
    basis[i] = 0.0f;
  }
  basis[points_len + order - 1] = 0.0f;

  for (int i_order = 2; i_order <= order; i_order++) {
    if (i2 + i_order >= points_len + order) {
      i2 = points_len + order - 1 - i_order;
    }
    for (int i = i1; i <= i2; i++) {
      float new_basis = 0.0f;
      if (basis[i] != 0.0f) {
        new_basis += ((t - knots[i]) * basis[i]) / (knots[i + i_order - 1] - knots[i]);
      }

      if (basis[i + 1] != 0.0f) {
        new_basis += ((knots[i + i_order] - t) * basis[i + 1]) /
                     (knots[i + i_order] - knots[i + 1]);
      }

      basis[i] = new_basis;
    }
  }

  start = 1000;
  end = 0;

  for (int i = i1; i <= i2; i++) {
    if (basis[i] > 0.0f) {
      end = i;
      if (start == 1000) {
        start = i;
      }
    }
  }
}

void NURBSpline::calculate_weights() const
{
  if (!this->weights_dirty_) {
    return;
  }

  std::lock_guard lock{this->weights_mutex_};
  if (!this->weights_dirty_) {
    return;
  }

  const int evaluated_len = this->evaluated_points_size();
  this->weight_cache_.resize(evaluated_len);

  const int points_len = this->size();
  const int order = this->order();
  Span<float> control_weights = this->weights();
  Span<float> knots = this->knots();

  MutableSpan<NURBSpline::WeightCache> weights = this->weight_cache_;

  const float start = knots[order - 1];
  const float end = this->is_cyclic ? knots[points_len + order - 1] : knots[points_len];
  const float step = (end - start) / (evaluated_len - (this->is_cyclic ? 0 : 1));

  Array<float> sums(points_len);
  Array<float> basis(this->knots_size());

  float u = start;
  for (const int i : IndexRange(evaluated_len)) {
    int j_start;
    int j_end;
    nurb_basis(
        u, points_len + (this->is_cyclic ? order - 1 : 0), order, knots, basis, j_start, j_end);
    BLI_assert(j_end - j_start < order);

    /* Calculate sums. */
    float sum_total = 0.0f;
    for (const int j : IndexRange(j_end - j_start + 1)) {
      const int point_index = (j_start + j) % points_len;

      sums[j] = basis[j_start + j] * control_weights[point_index];
      sum_total += sums[j];
    }
    if (sum_total != 0.0f) {
      for (const int j : IndexRange(j_end - j_start + 1)) {
        sums[j] /= sum_total;
      }
    }

    weights[i].start_index = j_start;
    weights[i].weights.clear();
    for (const int j : IndexRange(j_end - j_start + 1)) {
      weights[i].weights.append(sums[j]);
    }

    u += step;
  }

  this->weights_dirty_ = false;
}

template<typename T>
void interpolate_to_evaluated_points_impl(Span<NURBSpline::WeightCache> weights,
                                          const blender::VArray<T> &old_values,
                                          MutableSpan<T> r_values)
{
  const int points_len = old_values.size();
  BLI_assert(r_values.size() == weights.size());
  blender::attribute_math::DefaultMixer<T> mixer(r_values);

  for (const int i : r_values.index_range()) {
    Span<float> point_weights = weights[i].weights;
    const int start_index = weights[i].start_index;

    for (const int j : IndexRange(point_weights.size())) {
      const int point_index = (start_index + j) % points_len;
      mixer.mix_in(i, old_values[point_index], point_weights[j]);
    }
  }

  mixer.finalize();
}

blender::fn::GVArrayPtr NURBSpline::interpolate_to_evaluated_points(
    const blender::fn::GVArray &source_data) const
{
  this->calculate_weights();
  Span<WeightCache> weights = this->weight_cache_;

  blender::fn::GVArrayPtr new_varray;
  blender::attribute_math::convert_to_static_type(source_data.type(), [&](auto dummy) {
    using T = decltype(dummy);
    if constexpr (!std::is_void_v<blender::attribute_math::DefaultMixer<T>>) {
      Array<T> values(this->evaluated_points_size());
      interpolate_to_evaluated_points_impl<T>(weights, source_data.typed<T>(), values);
      new_varray = std::make_unique<blender::fn::GVArray_For_ArrayContainer<Array<T>>>(
          std::move(values));
    }
  });

  return new_varray;
}

void NURBSpline::ensure_base_cache() const
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

  blender::fn::GVArray_For_Span<float3> positions_varray(this->positions_.as_span());
  blender::fn::GVArrayPtr evaluated_positions_varray = this->interpolate_to_evaluated_points(
      positions_varray);

  Span<float3> evaluated_positions =
      evaluated_positions_varray->typed<float3>()->get_internal_span();

  for (const int i : IndexRange(total)) {
    this->evaluated_positions_cache_[i] = evaluated_positions[i];
  }

  this->base_cache_dirty_ = false;
}
