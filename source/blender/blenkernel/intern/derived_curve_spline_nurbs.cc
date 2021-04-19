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

#include "BKE_curve.h"
#include "BKE_derived_curve.hh"

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
