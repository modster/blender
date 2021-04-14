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

/** \file
 * \ingroup eevee
 *
 * Shading passes contain drawcalls specific to shading pipelines.
 * They are to be shared across views.
 * This file is only for shading passes. Other passes are declared in their own module.
 */

#pragma once

#include "DRW_render.h"

#include "eevee_velocity.hh"

namespace blender::eevee {

class Instance;

/* -------------------------------------------------------------------- */
/** \name Passes
 * \{ */

class ForwardPass {
 private:
  Instance &inst_;

  DRWPass *opaque_ps_ = nullptr;
  DRWPass *light_additional_ps_ = nullptr;

 public:
  ForwardPass(Instance &inst) : inst_(inst){};

  void sync(void);
  void surface_add(Object *ob, Material *mat, int matslot);
  void render(void);
};

/** \} */

/* -------------------------------------------------------------------- */
/** \name ShadingPasses
 *
 * \{ */

/**
 * Shading passes. Shared between views. Objects will subscribe to one of them.
 */
class ShadingPasses {
 public:
  // BackgroundShadingPass background;
  // DeferredPass opaque;
  ForwardPass opaque;
  VelocityPass velocity;

 public:
  ShadingPasses(Instance &inst) : opaque(inst), velocity(inst){};

  void sync()
  {
    opaque.sync();
    velocity.sync();
  }
};

/** \} */

}  // namespace blender::eevee