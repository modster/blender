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
 * Templated wrappers to make it easier to use GPU objects in C++.
 */

#pragma once

#include "DRW_gpu_wrapper.hh"

namespace blender::eevee {

static inline void shgroup_geometry_call(DRWShadingGroup *grp,
                                         Object *ob,
                                         GPUBatch *geom,
                                         int v_first = -1,
                                         int v_count = -1,
                                         bool use_instancing = false)
{
  if (grp == nullptr) {
    return;
  }

  if (v_first == -1) {
    DRW_shgroup_call(grp, geom, ob);
  }
  else if (use_instancing) {
    DRW_shgroup_call_instance_range(grp, ob, geom, v_first, v_count);
  }
  else {
    DRW_shgroup_call_range(grp, ob, geom, v_first, v_count);
  }
}

}  // namespace blender::eevee
