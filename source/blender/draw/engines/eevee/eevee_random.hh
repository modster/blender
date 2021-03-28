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
 * Random Number Generator
 */

#pragma once

#include "GPU_framebuffer.h"
#include "GPU_texture.h"

#include "DRW_render.h"

typedef struct EEVEE_Random {
  /** 1 based current sample. */
  uint64_t sample = 1;
  /** Target sample count. */
  uint64_t sample_count = 1;

  void reset(void)
  {
    sample = 1;
  }

  /* Return true if a new iteration is needed. */
  bool step(void)
  {
    if (sample <= sample_count) {
      sample++;
      return true;
    }
    return false;
  }
} EEVEE_Random;