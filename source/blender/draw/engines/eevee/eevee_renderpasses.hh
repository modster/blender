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
 *
 * Copyright 2021, Blender Foundation.
 */

#pragma once

#include "eevee_accumulator.hh"

using namespace blender;

enum eEEVEERenderPassBit {
  NONE = 0,
  COMBINED = (1 << 0),
  DEPTH = (1 << 1),
  NORMAL = (1 << 2),
};

ENUM_OPERATORS(eEEVEERenderPassBit, NORMAL)

typedef struct EEVEE_RenderPasses {
 public:
  EEVEE_Accumulator *combined = nullptr;
  EEVEE_Accumulator *depth = nullptr;
  EEVEE_Accumulator *normal = nullptr;
  Vector<EEVEE_Accumulator *> aovs;

 private:
  EEVEE_Shaders &shaders_;
  eEEVEERenderPassBit enabled_passes_ = NONE;

 public:
  EEVEE_RenderPasses(EEVEE_Shaders &shaders) : shaders_(shaders){};

  ~EEVEE_RenderPasses()
  {
    delete combined;
    delete depth;
    delete normal;
  }

  void configure(eEEVEERenderPassBit passes, EEVEE_AccumulatorParameters &accum_params)
  {
#define PASS_CONFIGURE(pass_, enum_, format_) \
  this->pass_configure(accum_params, pass_, (passes & enum_) != 0, STRINGIFY(pass_), format_)

    PASS_CONFIGURE(combined, COMBINED, GPU_RGBA16F);
    PASS_CONFIGURE(depth, DEPTH, GPU_R16F);
    PASS_CONFIGURE(normal, NORMAL, GPU_RGBA16F);

#undef PASS_CONFIGURE

    enabled_passes_ = passes;
  }

  void init(void)
  {
    if (combined) {
      combined->init();
    }
    if (depth) {
      depth->init();
    }
    if (normal) {
      normal->init();
    }
    for (EEVEE_Accumulator *aov : aovs) {
      aov->init();
    }
  }

  eEEVEERenderPassBit enabled_passes_get(void)
  {
    return enabled_passes_;
  }

 private:
  void pass_configure(EEVEE_AccumulatorParameters &accum_params,
                      EEVEE_Accumulator *&pass,
                      bool enable,
                      const char *name,
                      eGPUTextureFormat format)
  {
    if (enable && pass && pass->parameters != accum_params) {
      /* Parameters have changed, need to reconstruct the accumulator. */
      delete pass;
      pass = nullptr;
    }

    if (enable && pass == nullptr) {
      pass = new EEVEE_Accumulator(shaders_, name, format, accum_params);
    }
    else if (!enable && pass != nullptr) {
      /* Delete unused passes. */
      delete pass;
      pass = nullptr;
    }
  }

} EEVEE_RenderPasses;