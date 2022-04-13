/* SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup bke
 */

#include "BKE_curves.hh"

#include "testing/testing.h"

namespace blender::bke::gpencil {

class GPencilFrame : public CurvesGeometry {
 public:
  GPencilFrame(){};
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

class GPDataRuntime {
 public:
  /* Runtime Data */
  /* void *stroke_painting_buffer; */
};

class GPencilData : blender::dna::gpencil::GPData {
 public:
  GPencilData();
  ~GPencilData();
};

}  // namespace blender::bke::gpencil

namespace blender::dna::gpencil {

#ifdef __cplusplus
class GPDataRuntime;
using GPDataRuntimeHandle = blender::bke::gpencil::GPDataRuntime;
#else
typedef struct GPDataRuntimeHandle GPDataRuntimeHandle;
#endif

typedef struct GPLayerGroup {
  struct GPLayerGroup *children;
  int children_size;

  int *layer_indices;
  int layer_indices_size;

  char name[128];
} GPLayerGroup;

typedef struct GPLayer {
  char name[128];
  int flag;
} GPLayer;

typedef struct GPData {
  CurvesGeometry *frames;
  int frames_size;
  CustomData frame_data;
  int active_frame_index;

  GPLayer *layers;
  int layers_size;
  CustomData layer_data;
  int active_layer_index;

  GPLayerGroup default_group;

  GPDataRuntimeHandle *runtime;
} GPData;

typedef struct GreasePencil {
  ID id;
  /* Animation data (must be immediately after id). */
  struct AnimData *adt;

  /**/
  GPData grease_pencil_data;

  int flag;

  /** Materials array. */
  struct Material **mat;
  /** Total materials. */
  short totcol;

  /* ... */
} GreasePencil;

}  // namespace blender::dna::gpencil

namespace blender::bke::gpencil::tests {

TEST(gpencil_proposal, Foo)
{
  using namespace blender::bke::gpencil;
  GPencilFrame my_frame;
  float3 min, max;
  EXPECT_FALSE(my_frame.bounds_min_max(min, max));
}

}  // namespace blender::bke::gpencil::tests