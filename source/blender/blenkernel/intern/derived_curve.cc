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

static BezierHandleType handle_type_from_dna_bezt(const eBezTriple_Handle dna_handle_type)
{
  switch (dna_handle_type) {
    case HD_FREE:
      return BezierHandleType::Free;
    case HD_AUTO:
      return BezierHandleType::Auto;
    case HD_VECT:
      return BezierHandleType::Vector;
    case HD_ALIGN:
      return BezierHandleType::Align;
    case HD_AUTO_ANIM:
      return BezierHandleType::Auto;
    case HD_ALIGN_DOUBLESIDE:
      return BezierHandleType::Align;
  }
  BLI_assert_unreachable();
  return BezierHandleType::Free;
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
        ControlPointBezier point;
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

      curve->splines.append(spline);
    }
    else if (nurb->type == CU_NURBS) {
    }
    else if (nurb->type == CU_POLY) {
    }
  }

  return curve;
}

static void evaluate_bezier_part_3d(const float3 point_0,
                                    const float3 point_1,
                                    const float3 point_2,
                                    const float3 point_3,
                                    MutableSpan<float3> result)
{
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

int BezierSpline::evaluated_points_size() const
{
  BLI_assert(control_points.size() > 0);

  int total_len = 1;
  for (const int i : IndexRange(1, this->control_points.size() - 1)) {
    const ControlPointBezier &point_prev = this->control_points[i - 1];
    const ControlPointBezier &point = this->control_points[i];
    if (point_prev.handle_type_b == BezierHandleType::Vector &&
        point.handle_type_a == BezierHandleType::Vector) {
      total_len += 1;
    }
    else {
      total_len += this->resolution_u;
    }
  }

  return total_len;
}

void BezierSpline::ensure_evaluation_cache() const
{
  if (!this->cache_dirty) {
    return;
  }

  BezierSpline *mutable_self = const_cast<BezierSpline *>(this);
  std::lock_guard<std::mutex> lock(mutable_self->cache_mutex);
  mutable_self->evaluated_spline_cache.clear();

  mutable_self->evaluated_spline_cache.resize(this->evaluated_points_size());

  MutableSpan<float3> positions(mutable_self->evaluated_spline_cache);

  int offset = 0;
  for (const int i : IndexRange(1, this->control_points.size() - 1)) {
    const ControlPointBezier &point_prev = this->control_points[i - 1];
    const ControlPointBezier &point = this->control_points[i];

    if (point_prev.handle_type_b == BezierHandleType::Vector &&
        point.handle_type_a == BezierHandleType::Vector) {
      offset++;
    }
    else {
      const int resolution = this->resolution_u;
      evaluate_bezier_part_3d(point_prev.position,
                              point_prev.handle_position_b,
                              point.handle_position_a,
                              point.position,
                              positions.slice(offset, resolution));
      offset += resolution;
    }
  }

  positions.last() = control_points.last().position;

  mutable_self->cache_dirty = false;
}
