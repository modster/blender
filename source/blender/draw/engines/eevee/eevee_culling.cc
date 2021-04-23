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
 * A culling object is a data structure that contains fine grained culling
 * of entities against in the whole view frustum. The Culling structure contains the
 * final entity list since it has to have a special order.
 *
 * Follows the principles of Tiled Culling + Z binning from:
 * "Improved Culling for Tiled and Clustered Rendering"
 * by Michal Drobot
 * http://advances.realtimerendering.com/s2017/2017_Sig_Improved_Culling_final.pdf
 */

#include "eevee_instance.hh"

#include "eevee_culling.hh"

namespace blender::eevee {

/* -------------------------------------------------------------------- */
/** \name CullingDebugPass
 * \{ */

void CullingDebugPass::sync(void)
{
  LightModule &lights = inst_.lights;

  debug_ps_ = DRW_pass_create("CullingDebug", DRW_STATE_WRITE_COLOR);

  GPUShader *sh = inst_.shaders.static_shader_get(CULLING_DEBUG);
  DRWShadingGroup *grp = DRW_shgroup_create(sh, debug_ps_);
  DRW_shgroup_uniform_block_ref(grp, "lights_block", lights.data_ubo_ref_get());
  DRW_shgroup_uniform_block_ref(grp, "lights_culling_block", lights.culling_ubo_ref_get());
  DRW_shgroup_uniform_texture_ref(grp, "lights_culling_tx", lights.culling_tx_ref_get());
  DRW_shgroup_uniform_texture_ref(grp, "depth_tx", &input_depth_tx_);
  DRW_shgroup_call_procedural_triangles(grp, nullptr, 1);
}

void CullingDebugPass::render(GPUTexture *input_depth_tx)
{
  input_depth_tx_ = input_depth_tx;

  inst_.lights.bind_batch(0);

  DRW_draw_pass(debug_ps_);
}

/** \} */

}  // namespace blender::eevee