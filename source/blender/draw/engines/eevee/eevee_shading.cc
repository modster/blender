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

#include "eevee_instance.hh"

#include "eevee_shading.hh"

namespace blender::eevee {

/* -------------------------------------------------------------------- */
/** \name Passes
 * \{ */

void ForwardPass::sync()
{
  DRWState state = DRW_STATE_WRITE_COLOR | DRW_STATE_WRITE_DEPTH | DRW_STATE_DEPTH_LESS;
  opaque_ps_ = DRW_pass_create("Forward", state);

  DRWState state_add = DRW_STATE_WRITE_COLOR | DRW_STATE_BLEND_ADD_FULL | DRW_STATE_DEPTH_EQUAL;
  light_additional_ps_ = DRW_pass_create_instance("ForwardAddLight", opaque_ps_, state_add);
}

void ForwardPass::surface_add(Object *ob, Material *mat, int matslot)
{
  (void)mat;
  (void)matslot;
  GPUBatch *geom = DRW_cache_object_surface_get(ob);
  if (geom == nullptr) {
    return;
  }

  GPUShader *sh = inst_.shaders.static_shader_get(MESH);
  DRWShadingGroup *grp = DRW_shgroup_create(sh, opaque_ps_);
  DRW_shgroup_uniform_block_ref(grp, "lights_block", inst_.lights.ubo_ref_get());
  DRW_shgroup_uniform_block_ref(grp, "cluster_block", inst_.lights.cluster_ubo_ref_get());
  DRW_shgroup_call(grp, geom, ob);
}

void ForwardPass::render(void)
{
  for (auto index : inst_.lights.index_range()) {
    inst_.lights.bind_range(index);

    DRW_draw_pass((index == 0) ? opaque_ps_ : light_additional_ps_);
  }
}

/** \} */

}  // namespace blender::eevee