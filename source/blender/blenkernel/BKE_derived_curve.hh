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

class Spline {
 public:
  enum Type {
    Bezier,
    NURBS,
    Poly,
  };
  Type type;

  virtual int size() const = 0;
  virtual int resolution() const = 0;
  virtual void set_resolution(const int value) = 0;
  virtual void mark_cache_invalid() = 0;

  virtual int evaluated_points_size() const = 0;
  virtual void ensure_evaluation_cache() const = 0;
  virtual blender::Span<blender::float3> evaluated_positions() const = 0;
};

class BezierSpline : public Spline {
 public:
  /* TODO: Figure out if I want to store this as a few separate vectors directly in the spline. */
  blender::Vector<ControlPointBezier> control_points;
  int resolution_u;

  static constexpr inline Type static_type = Spline::Type::Bezier;

 private:
  bool cache_dirty;

  int32_t flag; /* Cyclic, smooth. */

  std::mutex cache_mutex;
  blender::Vector<blender::float3> evaluated_spline_cache;

 public:
  int size() const final
  {
    return control_points.size();
  }

  int resolution() const final
  {
    return resolution_u;
  }
  void set_resolution(const int value) final
  {
    resolution_u = value;
  }

  void mark_cache_invalid() final
  {
    cache_dirty = true;
  }

  int evaluated_points_size() const final;
  void ensure_evaluation_cache() const final;

  blender::Span<blender::float3> evaluated_positions() const final
  {
    this->ensure_evaluation_cache();
    return evaluated_spline_cache;
  }

  ~BezierSpline() = default;
};

class SplineNURBS : public Spline {
 public:
  blender::Vector<ControlPointNURBS> control_points;
  int32_t flag; /* Cyclic, smooth. */
  int resolution_u;
  uint8_t order;

  int size() const final
  {
    return control_points.size();
  }

  int resolution() const final
  {
    return resolution_u;
  }
  void set_resolution(const int value) final
  {
    resolution_u = value;
  }

  int evaluated_points_size() const final
  {
    return 0;
  }
  void ensure_evaluation_cache() const final
  {
  }

  blender::Span<blender::float3> evaluated_positions() const final
  {
    return {};
  }
};

/* Proposed name to be different from DNA type. */
struct DCurve {
  blender::Vector<Spline *> splines;
  int32_t flag; /* 2D. */

  /* Attributes. */
  //   AttributeStorage attributes;
  //   CustomData *control_point_data;
  //   CustomData *spline_data;
};

DCurve *dcurve_from_dna_curve(const Curve &curve);