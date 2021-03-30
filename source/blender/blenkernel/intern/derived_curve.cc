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

DCurve DCurve::from_dna_curve(const Curve &dna_curve)
{
  DCurve curve;

  curve.splines.reserve(BLI_listbase_count(&dna_curve.nurb));

  LISTBASE_FOREACH (const Nurb *, nurb, &dna_curve.nurb) {
    if (nurb->type == CU_BEZIER) {
      SplineBezier spline;
      for (const BezTriple &bezt : Span(nurb->bezt, nurb->pntsu)) {
        ControlPointBezier point;
        point.handle_position_a = bezt.vec[0];
        point.position = bezt.vec[1];
        point.handle_position_b = bezt.vec[2];
        point.radius = bezt.radius;
        point.tilt = bezt.tilt;
        point.handle_type_a = handle_type_from_dna_bezt((eBezTriple_Handle)bezt.h1);
        point.handle_type_b = handle_type_from_dna_bezt((eBezTriple_Handle)bezt.h2);
        spline.control_points.append(std::move(point));
      }

      spline.resolution_u = nurb->resolu;
      spline.resolution_v = nurb->resolv;
      spline.type = SplineType::Bezier;

      curve.splines.append(spline);
    }
    else if (nurb->type == CU_NURBS) {
    }
    else if (nurb->type == CU_POLY) {
    }
  }

  return curve;
}

void DCurve::ensure_evaluation_cache()
{
  this->evaluated_spline_cache.clear();

  for (Spline &spline : this->splines) {
    if (spline.type == SplineType::Bezier) {
      SplineBezier &spline_bezier = reinterpret_cast<SplineBezier &>(spline);
      for (ControlPointBezier &point : spline_bezier.control_points) {
        float3 *data = this->evaluated_spline_cache.end();
        float *data_axis = (float *)data;
        this->evaluated_spline_cache.reserve(this->evaluated_spline_cache.size() +
                                             spline_bezier.resolution_u);
        for (const int axis : {0, 1, 2}) {
          BKE_curve_forward_diff_bezier(point.position[axis],
                                        point.handle_position_b[axis],
                                        point.handle_position_a[axis],
                                        point.position[axis],
                                        data_axis + axis,
                                        spline_bezier.resolution_u,
                                        sizeof(float3));
        }
      }
    }
  }
}
