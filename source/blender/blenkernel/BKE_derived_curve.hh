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
#include "BLI_float4x4.hh"
#include "BLI_vector.hh"

#include "BKE_attribute_math.hh"

struct Curve;

/**
 * Contains the information necessary to interpolate values from the original control points to
 * evaluated points.
 */
struct PointMapping {
  /**
   * The index of the control point preceding the evaluated point at this index.
   * For resolutions higher than 1, many evaluated points can share the same value.
   */
  int control_point_index;
  /**
   * This evaluated point's portion of the total length between the `control_point_index` value
   * and the next.
   */
  float factor;
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
  using SplinePtr = std::unique_ptr<Spline>;

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
  Spline() = default;
  Spline(Spline &other)
      : type(other.type), is_cyclic(other.is_cyclic), normal_mode(other.normal_mode)
  {
    if (!other.base_cache_dirty_) {
      evaluated_positions_cache_ = other.evaluated_positions_cache_;
      evaluated_mapping_cache_ = other.evaluated_mapping_cache_;
      base_cache_dirty_ = false;
    }
    if (!other.tangent_cache_dirty_) {
      evaluated_tangents_cache_ = other.evaluated_tangents_cache_;
      tangent_cache_dirty_ = false;
    }
    if (!other.normal_cache_dirty_) {
      evaluated_normals_cache_ = other.evaluated_normals_cache_;
      normal_cache_dirty_ = false;
    }
    if (!other.length_cache_dirty_) {
      evaluated_lengths_cache_ = other.evaluated_lengths_cache_;
      length_cache_dirty_ = false;
    }
  }

  virtual SplinePtr copy() const = 0;

  virtual int size() const = 0;
  virtual int resolution() const = 0;
  virtual void set_resolution(const int value) = 0;

  virtual void drop_front(const int count) = 0;
  virtual void drop_back(const int count) = 0;

  virtual blender::MutableSpan<blender::float3> positions() = 0;
  virtual blender::Span<blender::float3> positions() const = 0;
  virtual blender::MutableSpan<float> radii() = 0;
  virtual blender::Span<float> radii() const = 0;
  virtual blender::MutableSpan<float> tilts() = 0;
  virtual blender::Span<float> tilts() const = 0;

  virtual void mark_cache_invalid();
  virtual int evaluated_points_size() const = 0;
  int evaluated_edges_size() const;

  float length() const;

  blender::Span<blender::float3> evaluated_positions() const;
  blender::Span<PointMapping> evaluated_mappings() const;
  blender::Span<float> evaluated_lengths() const;
  blender::Span<blender::float3> evaluated_tangents() const;
  blender::Span<blender::float3> evaluated_normals() const;

  struct LookupResult {
    /*
     * The index of the evaluated point before the result location.
     * In other words, the index of the edge that the result lies on.
     */
    int evaluated_index;
    /**
     * The portion of the way from the evaluated point at #index to the next point.
     */
    float factor;
  };
  LookupResult lookup_evaluated_factor(const float factor) const;
  LookupResult lookup_evaluated_length(const float length) const;

  /**
   * Interpolate data from the original control points to the corresponding ealuated points.
   * \param source_data: Should have the same size as the number of control points.
   * \param result_data: ...
   */
  template<typename T>
  void interpolate_data_to_evaluated_points(blender::Span<T> source_data,
                                            blender::MutableSpan<T> result_data,
                                            const int offset = 0) const
  {
    /* TODO: Onstead of "offset", it may be better to split a function that returns a single value.
     * TODO: Check for null default mixer, possibly using std::enable_if? */
    const int control_points_len = this->size();
    blender::Span<PointMapping> mappings = this->evaluated_mappings();

    blender::attribute_math::DefaultMixer<T> mixer(result_data);

    for (const int i : result_data.index_range()) {
      const int evaluated_point_index = offset + i;
      const PointMapping &mapping = mappings[evaluated_point_index];
      const int index = mapping.control_point_index;
      const int next_index = (index + 1) % control_points_len;
      const float factor = mapping.factor;

      const T &value = source_data[index];
      const T &next_value = source_data[next_index];

      mixer.mix_in(i, value, 1.0f - factor);
      mixer.mix_in(i, next_value, factor);
    }

    mixer.finalize();
  }

 protected:
  virtual void correct_end_tangents() const = 0;
  virtual void ensure_base_cache() const = 0;
};

using SplinePtr = std::unique_ptr<Spline>;

class BezierSpline final : public Spline {
 public:
  enum HandleType {
    Free,
    Auto,
    Vector,
    Align,
  };

 private:
  blender::Vector<HandleType> handle_types_start_;
  blender::Vector<blender::float3> handle_positions_start_;
  blender::Vector<blender::float3> positions_;
  blender::Vector<HandleType> handle_types_end_;
  blender::Vector<blender::float3> handle_positions_end_;
  blender::Vector<float> radii_;
  blender::Vector<float> tilts_;
  int resolution_u_;

 public:
  virtual SplinePtr copy() const final;
  BezierSpline() = default;
  BezierSpline(const BezierSpline &other)
      : Spline((Spline &)other),
        handle_types_start_(other.handle_types_start_),
        handle_positions_start_(other.handle_positions_start_),
        positions_(other.positions_),
        handle_types_end_(other.handle_types_end_),
        handle_positions_end_(other.handle_positions_end_),
        radii_(other.radii_),
        tilts_(other.tilts_),
        resolution_u_(other.resolution_u_)
  {
  }

  int size() const final;
  int resolution() const final;
  void set_resolution(const int value) final;

  blender::MutableSpan<blender::float3> positions() final;
  blender::Span<blender::float3> positions() const final;
  blender::MutableSpan<float> radii() final;
  blender::Span<float> radii() const final;
  blender::MutableSpan<float> tilts() final;
  blender::Span<float> tilts() const final;

  blender::Span<HandleType> handle_types_start() const;
  blender::MutableSpan<HandleType> handle_types_start();
  blender::Span<blender::float3> handle_positions_start() const;
  blender::MutableSpan<blender::float3> handle_positions_start();
  blender::Span<HandleType> handle_types_end() const;
  blender::MutableSpan<HandleType> handle_types_end();
  blender::Span<blender::float3> handle_positions_end() const;
  blender::MutableSpan<blender::float3> handle_positions_end();

  void add_point(const blender::float3 position,
                 const HandleType handle_type_start,
                 const blender::float3 handle_position_start,
                 const HandleType handle_type_end,
                 const blender::float3 handle_position_end,
                 const float radius,
                 const float tilt);

  void drop_front(const int count) final;
  void drop_back(const int count) final;

  int evaluated_points_size() const final;

  bool point_is_sharp(const int index) const;
  bool handle_start_is_automatic(const int index) const;
  bool handle_end_is_automatic(const int index) const;

  void move_control_point(const int index, const blender::float3 new_position);

 protected:
  void correct_final_tangents() const;

 private:
  void correct_end_tangents() const final;
  void ensure_base_cache() const final;
  bool segment_is_vector(const int start_index) const;
  void evaluate_bezier_segment(const int index,
                               const int next_index,
                               int &offset,
                               blender::MutableSpan<blender::float3> positions,
                               blender::MutableSpan<PointMapping> mappings) const;
  void evaluate_bezier_position_and_mapping(blender::MutableSpan<blender::float3> positions,
                                            blender::MutableSpan<PointMapping> mappings) const;
};

class NURBSpline final : public Spline {
 private:
  blender::Vector<blender::float3> positions_;
  blender::Vector<float> radii_;
  blender::Vector<float> tilts_;
  blender::Vector<float> weights_;
  int resolution_u_;
  uint8_t order_;

 public:
  SplinePtr copy() const final;
  NURBSpline() = default;
  NURBSpline(const NURBSpline &other)
      : Spline((Spline &)other),
        positions_(other.positions_),
        radii_(other.radii_),
        tilts_(other.tilts_),
        weights_(other.weights_),
        resolution_u_(other.resolution_u_),
        order_(other.order_)
  {
  }

  int size() const final;
  int resolution() const final;
  void set_resolution(const int value) final;
  uint8_t order() const;
  void set_order(const uint8_t value);

  blender::MutableSpan<blender::float3> positions() final;
  blender::Span<blender::float3> positions() const final;
  blender::MutableSpan<float> radii() final;
  blender::Span<float> radii() const final;
  blender::MutableSpan<float> tilts() final;
  blender::Span<float> tilts() const final;

  blender::MutableSpan<float> weights();
  blender::Span<float> weights() const;

  void add_point(const blender::float3 position,
                 const float radius,
                 const float tilt,
                 const float weight);

  void drop_front(const int count) final;
  void drop_back(const int count) final;

  int evaluated_points_size() const final;

 protected:
  void correct_end_tangents() const final;
  void ensure_base_cache() const final;
};

class PolySpline final : public Spline {
 public:
  blender::Vector<blender::float3> positions_;
  blender::Vector<float> radii_;
  blender::Vector<float> tilts_;

 private:
 public:
  SplinePtr copy() const final;
  PolySpline() = default;
  PolySpline(const PolySpline &other)
      : Spline((Spline &)other),
        positions_(other.positions_),
        radii_(other.radii_),
        tilts_(other.tilts_)
  {
  }

  int size() const final;
  int resolution() const final;
  void set_resolution(const int value) final;

  blender::MutableSpan<blender::float3> positions() final;
  blender::Span<blender::float3> positions() const final;
  blender::MutableSpan<float> radii() final;
  blender::Span<float> radii() const final;
  blender::MutableSpan<float> tilts() final;
  blender::Span<float> tilts() const final;

  void add_point(const blender::float3 position, const float radius, const float tilt);

  void drop_front(const int count) final;
  void drop_back(const int count) final;

  int evaluated_points_size() const final;

 protected:
  void correct_end_tangents() const final;
  void ensure_base_cache() const final;
};

/* Proposed name to be different from DNA type. */
class DCurve {
 public:
  blender::Vector<SplinePtr> splines;

  // bool is_2d;

  DCurve *copy();

  // DCurve *copy();

  void translate(const blender::float3 translation);
  void transform(const blender::float4x4 &matrix);
};

DCurve *dcurve_from_dna_curve(const Curve &curve);