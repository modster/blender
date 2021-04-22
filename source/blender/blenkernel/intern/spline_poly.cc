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

#include "BKE_curve.h"
#include "BKE_spline.hh"

using blender::Array;
using blender::float3;
using blender::float4x4;
using blender::IndexRange;
using blender::MutableSpan;
using blender::Span;
using blender::Vector;

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

Span<float3> PolySpline::evaluated_positions() const
{
  return this->positions();
}

// static blender::fn::GVArrayPtr bad_hack_copy_varray(const blender::fn::GVArray &source_data)
// {
// }

/* TODO: This function is hacky.. how to deal with poly spline interpolation? */
blender::fn::GVArrayPtr PolySpline::interpolate_to_evaluated_points(
    const blender::fn::GVArray &source_data) const
{
  BLI_assert(source_data.size() == this->size());

  if (source_data.is_span()) {
    return std::make_unique<blender::fn::GVArray_For_GSpan>(source_data.get_internal_span());
  }
  // if (source_data.is_single()) {
  //   BUFFER_FOR_CPP_TYPE_VALUE(source_data.type(), value);
  //   source_data.get_internal_single(value);
  //   return std::make_unique<blender::fn::GVArray_For_SingleValue>(
  //       source_data.type(), source_data.size(), value);
  // }

  return {};
}