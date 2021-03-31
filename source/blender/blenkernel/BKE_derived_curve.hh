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

#pragma once

/** \file
 * \ingroup bke
 */

#include <mutex>

#include "BLI_float3.hh"
#include "BLI_vector.hh"

#include "BKE_curve.h"

struct Curve;

enum class BezierHandleType {
  Free,
  Auto,
  Vector,
  Align,
};

struct ControlPoint {
  blender::float3 position;
  float radius;
  /* User defined tilt in radians, added on top of the auto-calculated tilt. */
  float tilt;
};

struct ControlPointBezier : ControlPoint {
  blender::float3 handle_position_a;
  blender::float3 handle_position_b;
  BezierHandleType handle_type_a;
  BezierHandleType handle_type_b;
};

struct ControlPointNURBS : ControlPoint {
  blender::float3 position;
  float radius;
  float weight;
};

enum class SplineType {
  Bezier,
  Poly,
  NURBS,
};

struct Spline {
  SplineType type;

  virtual int size() const = 0;
};

struct SplineBezier : Spline {
  blender::Vector<ControlPointBezier> control_points;

  blender::Vector<blender::float3> handle_positions_a;
  blender::Vector<blender::float3> positions;
  blender::Vector<blender::float3> handle_positions_b;

  blender::Vector<BezierHandleType> handle_type_a;
  blender::Vector<BezierHandleType> handle_type_b;

  int32_t flag; /* Cyclic, smooth. */
  int32_t resolution_u;
  int32_t resolution_v;

  int size() const final
  {
    return control_points.size();
  }

  ~SplineBezier() = default;
};

struct SplineNURBS : Spline {
  blender::Vector<ControlPointNURBS> control_points;
  int32_t flag; /* Cyclic, smooth. */
  int32_t resolution_u;
  int32_t resolution_v;
  uint8_t order;

  int size() const final
  {
    return control_points.size();
  }
};

/* Proposed name to be different from DNA type. */
struct DCurve {
  blender::Vector<Spline *> splines;
  //   AttributeStorage attributes;
  int32_t flag; /* 2D. */

  /* Attributes. */
  //   CustomData *control_point_data;
  //   CustomData *spline_data;

  /* Then maybe whatever caches are necessary, etc. */
  //   std::mutex cache_mutex;
  blender::Vector<blender::float3> evaluated_spline_cache;

  void ensure_evaluation_cache();

  ~DCurve()
  {
    for (Spline *spline : splines) {
      if (spline->type == SplineType::Bezier) {
        SplineBezier *spline_bezier = reinterpret_cast<SplineBezier *>(spline);
        delete spline_bezier;
      }
    }
  }
};

DCurve *dcurve_from_dna_curve(const Curve &curve);