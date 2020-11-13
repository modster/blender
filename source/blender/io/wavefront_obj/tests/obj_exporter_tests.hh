/* Apache License, Version 2.0 */

/**
 * This file contains default values for several items like
 * vertex coordinates, export parameters, MTL values etc.
 */

#pragma once

#include <array>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace blender::io::obj {

using arr_float_3 = std::array<float, 3>;

class NurbsObject {
 private:
  std::string nurbs_name_;
  std::vector<std::vector<arr_float_3>> coordinates_;

 public:
  NurbsObject(const std::string nurbs_name,
              const std::vector<std::vector<arr_float_3>> coordinates)
      : nurbs_name_(nurbs_name), coordinates_(coordinates)
  {
  }

  int total_splines() const
  {
    return coordinates_.size();
  }

  int total_nurbs_points(const int spline_index) const
  {
    if (spline_index >= coordinates_.size()) {
      ADD_FAILURE();
      return 0;
    }
    return coordinates_[spline_index].size();
  }

  const float *get_nurbs_point_coords(const int spline_index, const int vertex_index) const
  {
    return coordinates_[spline_index][vertex_index].data();
  }
};

const std::vector<std::vector<arr_float_3>> coordinates_NurbsCurve{
    {{9.947419, 0.000000, 0.000000},
     {9.447419, 0.000000, 1.000000},
     {7.447419, 0.000000, 1.000000},
     {6.947419, 0.000000, 0.000000}}};
const std::vector<std::vector<arr_float_3>> coordinates_NurbsCircle{
    {{11.463165, 0.000000, -1.000000},
     {12.463165, 0.000000, -1.000000},
     {12.463165, 0.000000, 0.000000},
     {12.463165, 0.000000, 1.000000},
     {11.463165, 0.000000, 1.000000},
     {10.463165, 0.000000, 1.000000},
     {10.463165, 0.000000, 0.000000},
     {10.463165, 0.000000, -1.000000}}};
const std::vector<std::vector<arr_float_3>> coordinates_NurbsPath{
    {{17.690557, 0.000000, 0.000000},
     {16.690557, 0.000000, 0.000000},
     {15.690557, 0.000000, 0.000000},
     {14.690557, 0.000000, 0.000000},
     {13.690557, 0.000000, 0.000000}},
    {{17.188307, 0.000000, 0.000000},
     {16.688307, 0.000000, 1.000000},
     {14.688307, 0.000000, 1.000000},
     {14.188307, 0.000000, 0.000000}}};
}  // namespace blender::io::obj
