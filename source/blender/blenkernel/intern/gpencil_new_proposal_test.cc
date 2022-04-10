/* SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup bke
 */

#include "BKE_curves.hh"

#include "testing/testing.h"

namespace blender::bke::gpencil {

typedef struct bGPdata bGPdata;

class GPencilFrame : public CurvesGeometry {
  public:
  GPencilFrame() {};
  ~GPencilFrame() = default;

  CurvesGeometry &as_curves_geometry()
  {
    CurvesGeometry *geometry = reinterpret_cast<CurvesGeometry *>(this);
    return *geometry;
  }

  bool bounds_min_max(float3 &min, float3 &max) 
  {
    return as_curves_geometry().bounds_min_max(min, max);
  }
};

// class GPencilData : bGPdata {
// public:
//   GPencilData();
//   ~GPencilData();
// };

struct bGPdata {
  ID id;
  /* Animation data (must be immediately after id). */
  struct AnimData *adt;

  CurvesGeometry *frames;
  int frames_size;
  CustomData frame_data;


  /** Materials array. */
  struct Material **mat;
  /** Total materials. */
  short totcol;
};

namespace tests {

TEST(gpencil_proposal, Foo) {
  using namespace blender::bke::gpencil;
  GPencilFrame my_frame;
  float3 min, max;
  EXPECT_FALSE(my_frame.bounds_min_max(min, max));
}

} // namespace tests
} // namespace blender::bke::gpencil