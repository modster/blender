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
 * The Hierarchical-Z buffer is texture containing a copy of the depth buffer with mipmaps.
 * Each mip contains the maximum depth of each 4 pixels on the upper level.
 * The size of the texture is padded to avoid messing with the mipmap pixels alignments.
 */

#include "eevee_instance.hh"

#include "eevee_hizbuffer.hh"

namespace blender::eevee {

/* -------------------------------------------------------------------- */
/** \name Hierarchical-Z buffer
 *
 * \{ */

void HiZBufferModule::sync(void)
{
  {
    hiz_copy_ps_ = DRW_pass_create("HizCopy", DRW_STATE_WRITE_COLOR);
    GPUShader *sh = inst_.shaders.static_shader_get(HIZ_COPY);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, hiz_copy_ps_);
    DRW_shgroup_uniform_texture_ref(grp, "depth_tx", &input_depth_tx_);
    DRW_shgroup_call_procedural_triangles(grp, nullptr, 1);
  }
  {
    hiz_downsample_ps_ = DRW_pass_create("HizDownsample", DRW_STATE_WRITE_COLOR);
    GPUShader *sh = inst_.shaders.static_shader_get(HIZ_DOWNSAMPLE);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, hiz_downsample_ps_);
    DRW_shgroup_uniform_texture_ref(grp, "depth_tx", &input_depth_tx_);
    DRW_shgroup_uniform_vec2(grp, "texel_size", texel_size_, 1);
    DRW_shgroup_call_procedural_triangles(grp, nullptr, 1);
  }
}

void HiZBuffer::prepare(GPUTexture *depth_src_tx)
{
  int div = 1 << mip_count_;
  vec2 extent_src(GPU_texture_width(depth_src_tx), GPU_texture_height(depth_src_tx));
  vec2 extent_hiz(divide_ceil_u(extent_src.x, div) * div, divide_ceil_u(extent_src.y, div) * div);

  inst_.hiz.data_.pixel_to_ndc = 2.0f / extent_src;
  inst_.hiz.texel_size_ = 1.0f / extent_hiz;
  inst_.hiz.data_.uv_scale = extent_src / extent_hiz;

  inst_.hiz.data_.push_update();

  /* TODO/OPTI(fclem): Share it between similar views.
   * Not possible right now because request_tmp does not support mipmaps. */
  hiz_tx_.ensure("HiZ", UNPACK2(extent_hiz), mip_count_, GPU_R32F);
  hiz_fb_.ensure(GPU_ATTACHMENT_NONE, GPU_ATTACHMENT_TEXTURE(hiz_tx_));

  GPU_texture_mipmap_mode(hiz_tx_, true, false);
}

void HiZBuffer::recursive_downsample(void *thunk, int UNUSED(lvl))
{
  HiZBufferModule &hiz = *reinterpret_cast<HiZBufferModule *>(thunk);
  hiz.texel_size_ *= 2.0f;
  DRW_draw_pass(hiz.hiz_downsample_ps_);
}

void HiZBuffer::update(GPUTexture *depth_src_tx)
{
  DRW_stats_group_start("Hiz");

  inst_.hiz.texel_size_ = 1.0f / vec2(GPU_texture_width(hiz_tx_), GPU_texture_height(hiz_tx_));

  inst_.hiz.input_depth_tx_ = depth_src_tx;
  GPU_framebuffer_bind(hiz_fb_);
  DRW_draw_pass(inst_.hiz.hiz_copy_ps_);

  inst_.hiz.input_depth_tx_ = hiz_tx_;
  GPU_framebuffer_recursive_downsample(hiz_fb_, mip_count_, &recursive_downsample, &inst_.hiz);

  DRW_stats_group_end();
}

/** \} */

}  // namespace blender::eevee
