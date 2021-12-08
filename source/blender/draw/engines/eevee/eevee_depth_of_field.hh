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
 * Depth of field post process effect.
 *
 * There are 2 methods to achieve this effect.
 * - The first uses projection matrix offsetting and sample accumulation to give
 * reference quality depth of field. But this needs many samples to hide the
 * under-sampling.
 * - The second one is a post-processing based one. It follows the
 * implementation described in the presentation "Life of a Bokeh - Siggraph
 * 2018" from Guillaume Abadie. There are some difference with our actual
 * implementation that prioritize quality.
 */

#pragma once

#include "eevee_shader_shared.hh"

namespace blender::eevee {

class Instance;

/* -------------------------------------------------------------------- */
/** \name Depth of field
 * \{ */

class DepthOfField {
 private:
  class Instance &inst_;

  DepthOfFieldDataBuf data_;

  /** Textures from pool. Not owned. */
  GPUTexture *bokeh_gather_lut_tx_ = nullptr;
  GPUTexture *bokeh_resolve_lut_tx_ = nullptr;
  GPUTexture *bokeh_scatter_lut_tx_ = nullptr;
  GPUTexture *color_bg_tx_ = nullptr;
  GPUTexture *color_fg_tx_ = nullptr;
  GPUTexture *color_holefill_tx_ = nullptr;
  GPUTexture *occlusion_tx_ = nullptr;
  GPUTexture *reduce_downsample_tx_ = nullptr;
  GPUTexture *scatter_src_tx_ = nullptr;
  GPUTexture *setup_coc_tx_ = nullptr;
  GPUTexture *setup_color_tx_ = nullptr;
  GPUTexture *tiles_bg_tx_ = nullptr;
  GPUTexture *tiles_fg_tx_ = nullptr;
  GPUTexture *tiles_dilated_bg_tx_ = nullptr;
  GPUTexture *tiles_dilated_fg_tx_ = nullptr;
  GPUTexture *weight_bg_tx_ = nullptr;
  GPUTexture *weight_fg_tx_ = nullptr;
  GPUTexture *weight_holefill_tx_ = nullptr;
  /** Allocated textures. Owned. */
  eevee::Texture reduced_coc_tx_ = Texture("dof_reduced_coc");
  eevee::Texture reduced_color_tx_ = Texture("dof_reduced_color");
  /** Input texture. Not owned. */
  GPUTexture *input_color_tx_;
  GPUTexture *input_depth_tx_;
  /** Passes. Not owned. */
  DRWPass *bokeh_lut_ps_ = nullptr;
  DRWPass *gather_bg_ps_ = nullptr;
  DRWPass *gather_fg_ps_ = nullptr;
  DRWPass *filter_ps_ = nullptr;
  DRWPass *gather_holefill_ps_ = nullptr;
  DRWPass *reduce_copy_ps_ = nullptr;
  DRWPass *reduce_downsample_ps_ = nullptr;
  DRWPass *reduce_recursive_ps_ = nullptr;
  DRWPass *resolve_ps_ = nullptr;
  DRWPass *scatter_bg_ps_ = nullptr;
  DRWPass *scatter_fg_ps_ = nullptr;
  DRWPass *setup_ps_ = nullptr;
  DRWPass *tiles_dilate_minabs_ps_ = nullptr;
  DRWPass *tiles_dilate_minmax_ps_ = nullptr;
  DRWPass *tiles_flatten_ps_ = nullptr;
  /** Framebuffers. Owned.  */
  eevee::Framebuffer bokeh_lut_fb_ = Framebuffer("bokeh_lut_fb_");
  eevee::Framebuffer filter_bg_fb_ = Framebuffer("filter_bg_fb_");
  eevee::Framebuffer filter_fg_fb_ = Framebuffer("filter_fg_fb_");
  eevee::Framebuffer gather_fb_ = Framebuffer("gather_fb_");
  eevee::Framebuffer gather_filter_bg_fb_ = Framebuffer("gather_filter_bg_fb_");
  eevee::Framebuffer gather_holefill_fb_ = Framebuffer("gather_holefill_fb_");
  eevee::Framebuffer reduce_copy_fb_ = Framebuffer("reduce_copy_fb_");
  eevee::Framebuffer reduce_downsample_fb_ = Framebuffer("reduce_downsample_fb_");
  eevee::Framebuffer reduce_fb_ = Framebuffer("reduce_fb_");
  eevee::Framebuffer resolve_fb_ = Framebuffer("resolve_fb_");
  eevee::Framebuffer scatter_bg_fb_ = Framebuffer("scatter_bg_fb_");
  eevee::Framebuffer scatter_fg_fb_ = Framebuffer("scatter_fg_fb_");
  eevee::Framebuffer setup_fb_ = Framebuffer("setup_fb_");
  eevee::Framebuffer tiles_dilate_fb_ = Framebuffer("tiles_dilate_fb_");
  eevee::Framebuffer tiles_flatten_fb_ = Framebuffer("tiles_flatten_fb_");

  /** Scene settings that are immutable. */
  float user_overblur_;
  float fx_max_coc_;
  /** Use Hiqh Quality (expensive) in-focus gather pass. */
  bool do_hq_slight_focus_;
  /** Use jittered depth of field where we randomize camera location. */
  bool do_jitter_;

  /** Circle of Confusion radius for FX DoF passes. Is in view X direction in [0..1] range. */
  float fx_radius_;
  /** Circle of Confusion radius for jittered DoF. Is in view X direction in [0..1] range. */
  float jitter_radius_;
  /** Focus distance in view space. */
  float focus_distance_;
  /** Extent of the input buffer. */
  ivec2 extent_;

  /** Tile dilation uniforms. */
  int tiles_dilate_slight_focus_;
  int tiles_dilate_ring_count_;
  int tiles_dilate_ring_width_multiplier_;

  /** Reduce pass info. */
  int reduce_steps_;

  /** Static string pointer. Used as debug name and as UUID for texture pool. */
  StringRefNull view_name_;

 public:
  DepthOfField(Instance &inst, StringRefNull view_name) : inst_(inst), view_name_(view_name){};
  ~DepthOfField(){};

  void init();

  void sync(const mat4 winmat, ivec2 input_extent);

  /** Apply Depth Of Field jittering to the view and projection matrices.. */
  void jitter_apply(mat4 winmat, mat4 viewmat);

  /** Will swap input and output texture if rendering happens. The actual output of this function
   * is in intput_tx. */
  void render(GPUTexture *depth_tx, GPUTexture **input_tx, GPUTexture **output_tx);

 private:
  void bokeh_lut_pass_sync(void);
  void bokeh_lut_pass_render(void);

  void setup_pass_sync(void);
  void setup_pass_render(void);

  void tiles_prepare_pass_sync(void);
  void tiles_prepare_pass_render(void);

  static void reduce_recusive(void *thunk, int level);
  void reduce_pass_sync(void);
  void reduce_pass_render(void);

  void convolve_pass_sync(void);
  void convolve_pass_render(void);

  void resolve_pass_sync(void);
  void resolve_pass_render(GPUTexture *output_tx);
};

/** \} */

}  // namespace blender::eevee