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

struct PointMapping {
  int control_point_index;
  /* Linear interpolation factor starting at this control point with the index in
   * `control_point_index`, and ending with the next control point. */
  float factor;
};

/* TODO: Think about storing each data type from each control point separately. */
struct BezierPoint {
  enum HandleType {
    Free,
    Auto,
    Vector,
    Align,
  };

  /* The first handle. */
  HandleType handle_type_a;
  blender::float3 handle_position_a;

  blender::float3 position;

  /* The second handle. */
  HandleType handle_type_b;
  blender::float3 handle_position_b;

  float radius;
  /* User defined tilt in radians, added on top of the auto-calculated tilt. */
  float tilt;

  bool is_sharp() const
  {
    return ELEM(handle_type_a, HandleType::Vector, HandleType::Free) ||
           ELEM(handle_type_b, HandleType::Vector, HandleType::Free);
  }
};

/**
 * A spline is an abstraction of a curve section, its evaluation methods, and data.
 * The spline data itself is just control points and a set of attributes.
 *
 * Common evaluated data is stored in caches on the spline itself. This way operations on splines
 * don't need to worry about taking ownership of evaluated data when they don't need to.
 */
class Spline {
 public:
  enum Type {
    Bezier,
    NURBS,
    Poly,
  };
  Type type;
  bool is_cyclic = false;

  enum NormalCalculationMode {
    ZUp,
    Minimum,
    Tangent,
  };
  NormalCalculationMode normal_mode;

 protected:
  mutable bool base_cache_dirty_ = true;
  mutable std::mutex base_cache_mutex_;
  mutable blender::Vector<blender::float3> evaluated_positions_cache_;
  mutable blender::Vector<PointMapping> evaluated_mapping_cache_;

  mutable bool tangent_cache_dirty_ = true;
  mutable std::mutex tangent_cache_mutex_;
  mutable blender::Vector<blender::float3> evaluated_tangents_cache_;

  mutable bool normal_cache_dirty_ = true;
  mutable std::mutex normal_cache_mutex_;
  mutable blender::Vector<blender::float3> evaluated_normals_cache_;

  mutable bool length_cache_dirty_ = true;
  mutable std::mutex length_cache_mutex_;
  mutable blender::Vector<float> evaluated_lengths_cache_;

 public:
  virtual ~Spline() = default;

  virtual int size() const = 0;
  virtual int resolution() const = 0;
  virtual void set_resolution(const int value) = 0;

  virtual void mark_cache_invalid();
  virtual int evaluated_points_size() const = 0;
  int evaluated_edges_size() const;

  blender::Span<blender::float3> evaluated_positions() const;
  blender::Span<PointMapping> evaluated_mappings() const;
  blender::Span<float> evaluated_lengths() const;
  blender::Span<blender::float3> evaluated_tangents() const;
  blender::Span<blender::float3> evaluated_normals() const;

  /* TODO: I'm not sure this is the best abstraction here, maybe we want another cache. */
  float get_evaluated_point_radius(const int index) const;

  void trim_lengths(const float start_length, const float end_length);

 protected:
  virtual void correct_end_tangents() const = 0;
  virtual void ensure_base_cache() const = 0;

  virtual float control_point_radius(const int index) const = 0;
};

class BezierSpline : public Spline {
 public:
  blender::Vector<BezierPoint> control_points;

 private:
  int resolution_u;

 public:
  ~BezierSpline() = default;

  int size() const final;
  int resolution() const final;
  void set_resolution(const int value) final;
  int evaluated_points_size() const final;

 protected:
  void correct_final_tangents() const;

 private:
  void correct_end_tangents() const final;
  void ensure_base_cache() const final;

  float control_point_radius(const int index) const final;
};

struct NURBSPoint {
  blender::float3 position;
  float radius;
  float weight;

  /* User defined tilt in radians, added on top of the auto-calculated tilt. */
  float tilt;
};

class NURBSPline : public Spline {
 public:
  blender::Vector<NURBSPoint> control_points;

 private:
  int32_t flag; /* Cyclic, smooth. */
  int resolution_u;
  uint8_t order;

 public:
  int size() const final;
  int resolution() const final;
  void set_resolution(const int value) final;
  int evaluated_points_size() const final;

 protected:
  void correct_end_tangents() const final;
  void ensure_base_cache() const final;

  float control_point_radius(const int index) const final;
};

/* Proposed name to be different from DNA type. */
struct DCurve {
  blender::Vector<Spline *> splines;

  // bool is_2d;

  // DCurve *copy();

  ~DCurve()
  {
    for (Spline *spline : splines) {
      delete spline;
    }
  }
};

DCurve *dcurve_from_dna_curve(const Curve &curve);