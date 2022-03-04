/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#pragma once

#include "BLI_math.h"
#include "BLI_math_vec_types.hh"
#include "BLI_sys_types.h"

namespace blender::imbuf::rasterizer {

/** How to integrate the result of a fragment shader into its drawing target. */
template<typename Source, typename Destination> class AbstractBlendMode {
 public:
  using SourceType = Source;
  using DestinationType = Destination;

  virtual void blend(Destination *dest, const Source &source) const = 0;
};

/**
 * Copy the result of the fragment shader into float[4] without any modifications.
 */
class CopyBlendMode : public AbstractBlendMode<float4, float> {
 public:
  void blend(float *dest, const float4 &source) const override
  {
    copy_v4_v4(dest, source);
  }
};

class AlphaBlendMode : public AbstractBlendMode<float4, float> {
 public:
  void blend(float *dest, const float4 &source) const override
  {
    interp_v3_v3v3(dest, dest, source, source[3]);
    dest[3] = 1.0;
  }
};

}  // namespace blender::imbuf::rasterizer
