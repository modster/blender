/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2021 Blender Foundation.
 */

/** \file
 * \ingroup eevee
 *
 * The Hierarchical-Z buffer is texture containing a copy of the depth buffer with mipmaps.
 * Each mip contains the maximum depth of each 4 pixels on the upper level.
 * The size of the texture is padded to avoid messing with the mipmap pixels alignments.
 */

#include "BKE_global.h"

#include "eevee_instance.hh"

#include "eevee_hizbuffer.hh"

namespace blender::eevee {

/* -------------------------------------------------------------------- */
/** \name Hierarchical-Z buffer
 *
 * \{ */

void HiZBuffer::begin_sync()
{
  extent_ = int2(-1);
}

void HiZBuffer::view_sync(int2 view_extent)
{
  extent_ = math::max(extent_, view_extent);
}

void HiZBuffer::end_sync()
{
  extent_.x = ceil_multiple_u(extent_.x, kernel_size_);
  extent_.y = ceil_multiple_u(extent_.y, kernel_size_);

  hiz_tx_.ensure_2d(GPU_R32F, extent_, nullptr, mip_count_);
  hiz_tx_.ensure_mip_views();
  GPU_texture_mipmap_mode(hiz_tx_, true, false);

  {
    hiz_update_ps_ = DRW_pass_create("HizUpdate", (DRWState)0);
    GPUShader *sh = inst_.shaders.static_shader_get(HIZ_UPDATE);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, hiz_update_ps_);
    DRW_shgroup_uniform_texture_ref_ex(grp, "depth_tx", &input_depth_tx_, GPU_SAMPLER_FILTER);
    DRW_shgroup_uniform_image(grp, "out_lvl0", hiz_tx_.mip_view(0));
    DRW_shgroup_uniform_image(grp, "out_lvl1", hiz_tx_.mip_view(1));
    DRW_shgroup_uniform_image(grp, "out_lvl2", hiz_tx_.mip_view(2));
    DRW_shgroup_uniform_image(grp, "out_lvl3", hiz_tx_.mip_view(3));
    DRW_shgroup_uniform_image(grp, "out_lvl4", hiz_tx_.mip_view(4));
    DRW_shgroup_uniform_image(grp, "out_lvl5", hiz_tx_.mip_view(5));
    DRW_shgroup_call_compute_ref(grp, dispatch_size_);
    DRW_shgroup_barrier(grp, GPU_BARRIER_TEXTURE_FETCH);
  }

  is_dirty_ = true;
}

void HiZBuffer::update(GPUTexture *depth_src_tx)
{
  if (!is_dirty_) {
    return;
  }
  int2 extent_src = {GPU_texture_width(depth_src_tx), GPU_texture_height(depth_src_tx)};

  BLI_assert(extent_src.x <= extent_.x);
  BLI_assert(extent_src.y <= extent_.y);

  dispatch_size_.x = divide_ceil_u(extent_src.x, kernel_size_);
  dispatch_size_.y = divide_ceil_u(extent_src.y, kernel_size_);
  dispatch_size_.z = 1;

  inst_.hiz.data_.uv_scale = float2(extent_src) / float2(extent_);
  inst_.hiz.data_.push_update();

  input_depth_tx_ = depth_src_tx;

  /* Bind another framebuffer in order to avoid triggering the feedback loop check.
   * This is safe because we only use compute shaders in this section of the code.
   * Ideally the check should be smarter. */
  GPUFrameBuffer *fb = GPU_framebuffer_active_get();
  if (G.debug & G_DEBUG_GPU) {
    GPU_framebuffer_restore();
  }

  DRW_draw_pass(hiz_update_ps_);

  if (G.debug & G_DEBUG_GPU) {
    GPU_framebuffer_bind(fb);
  }
}

/** \} */

}  // namespace blender::eevee
