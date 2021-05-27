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
 * Copyright 2018, Blender Foundation.
 */

/** \file
 * \ingroup eevee
 */

#pragma once

#include "MEM_guardedalloc.h"

#include "DNA_lightprobe_types.h"

namespace blender::eevee {

/**
 * Wrapper to blender lightcache structure.
 * Used to define methods for the light cache.
 **/
struct LightCache : public ::LightCache {
  LightCache()
  {
    memset(this, 0, sizeof(*this));
  }

  ~LightCache()
  {
    DRW_TEXTURE_FREE_SAFE(cube_tx.tex);
    MEM_SAFE_FREE(cube_tx.data);
    DRW_TEXTURE_FREE_SAFE(grid_tx.tex);
    MEM_SAFE_FREE(grid_tx.data);

    if (cube_mips) {
      for (int i = 0; i < mips_len; i++) {
        MEM_SAFE_FREE(cube_mips[i].data);
      }
      MEM_SAFE_FREE(cube_mips);
    }

    MEM_SAFE_FREE(cube_data);
    MEM_SAFE_FREE(grid_data);
  }

  bool validate(void) const;

  MEM_CXX_CLASS_ALLOC_FUNCS("EEVEE:LightCache")
};

}  // namespace blender::eevee
