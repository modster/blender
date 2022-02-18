/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2021 Blender Foundation.
 */

/** \file
 * \ingroup eevee
 *
 * Templated wrappers to make it easier to use GPU objects in C++.
 */

#pragma once

#include "DRW_gpu_wrapper.hh"

namespace blender::eevee {

/* Returns true if the value is different and was assigned */
template<typename T> inline bool assign_if_different(T &old_value, T new_value)
{
  if (old_value != new_value) {
    old_value = new_value;
    return true;
  }
  return false;
}

using draw::Framebuffer;
using draw::Texture;
using draw::TextureFromPool;

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
