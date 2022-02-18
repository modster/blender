/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#pragma once

/** \file
 * \ingroup imbuf
 *
 * Pixel clamping determines how the edges of geometry is clamped to pixels.
 */

#include "BLI_sys_types.h"

namespace blender::imbuf::rasterizer {

class AbstractPixelClampingMethod {
 public:
  virtual float distance_to_scanline_anchor(float y) const = 0;
  virtual float distance_to_column_anchor(float y) const = 0;
  virtual int scanline_for(float y) const = 0;
  virtual int column_for(float x) const = 0;
};

class CenterPixelClampingMethod : public AbstractPixelClampingMethod {
 public:
  float distance_to_scanline_anchor(float y) const override
  {
    return distance_to_anchor(y);
  }
  float distance_to_column_anchor(float x) const override
  {
    return distance_to_anchor(x);
  }

  int scanline_for(float y) const override
  {
    return this->round(y);
  }

  int column_for(float x) const override
  {
    return this->round(x);
  }

 private:
  float distance_to_anchor(float value) const
  {
    float fract = to_fract(value);
    float result;
    if (fract <= 0.5f) {
      result = 0.5f - fract;
    }
    else {
      result = 1.5f - fract;
    }
    BLI_assert(result >= 0.0f);
    BLI_assert(result < 1.0f);
    return result;
  }

  int round(float value) const
  {
    /* Cannot use std::round as it rounds away from 0. */
    float fract = to_fract(value);
    int result;

    if (fract > 0.5f) {
      result = ceilf(value);
    }
    else {
      result = floorf(value);
    }
    return result;
  }

  float to_fract(float value) const
  {
    return value - floor(value);
  }
};

}  // namespace blender::imbuf::rasterizer
