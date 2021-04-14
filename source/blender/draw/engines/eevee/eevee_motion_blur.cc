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
 */

#include "BLI_map.hh"
#include "DEG_depsgraph_query.h"

#include "eevee_instance.hh"
#include "eevee_sampling.hh"
#include "eevee_shader_shared.hh"
#include "eevee_velocity.hh"

namespace blender::eevee {

/* -------------------------------------------------------------------- */
/** \name MotionBlurModule
 *
 * \{ */

void MotionBlurModule::init(void)
{
  const Scene *scene = inst_.scene;

  enabled_ = (scene->eevee.flag & SCE_EEVEE_MOTION_BLUR_ENABLED) != 0;

  /* Viewport not supported for now. */
  if (!DRW_state_is_scene_render()) {
    enabled_ = false;
  }
  if (!enabled_) {
    motion_blur_fx_enabled_ = false;
    return;
  }

  /* Take into account the steps needed for fx motion blur. */
  int steps_count = max_ii(1, scene->eevee.motion_blur_steps) * 2 + 1;

  time_steps_.resize(steps_count);

  initial_frame_ = CFRA;
  initial_subframe_ = SUBFRA;
  frame_time_ = initial_frame_ + initial_subframe_;
  motion_blur_position_ = scene->eevee.motion_blur_position;
  motion_blur_shutter_ = scene->eevee.motion_blur_shutter;

  /* Without this there is the possibility of the curve table not being allocated. */
  BKE_curvemapping_changed((struct CurveMapping *)&scene->r.mblur_shutter_curve, false);

  Vector<float> cdf(CM_TABLE);
  Sampling::cdf_from_curvemapping(scene->r.mblur_shutter_curve, cdf);
  Sampling::cdf_invert(cdf, time_steps_);

  for (float &time : time_steps_) {
    time = this->shutter_time_to_scene_time(time);
  }

  motion_blur_fx_enabled_ = scene->eevee.motion_blur_max > 0.5f;
  step_id_ = 1;

  if (motion_blur_fx_enabled_) {
    /* A bit weird but we have to sync the first 2 steps here because the step()
     * function is only called after rendering a sample. */
    inst_.velocity.step_sync(VelocityModule::STEP_PREVIOUS, time_steps_[0]);
    inst_.velocity.step_sync(VelocityModule::STEP_NEXT, time_steps_[2]);
  }
  inst_.set_time(time_steps_[1]);
}

/* Runs after rendering a sample. */
void MotionBlurModule::step(void)
{
  if (!enabled_) {
    return;
  }
  else if (inst_.sampling.finished()) {
    /* Restore original frame number. This is because the render pipeline expects it. */
    RE_engine_frame_set(inst_.render, initial_frame_, initial_subframe_);
  }
  else if (inst_.sampling.do_render_sync()) {
    /* Time to change motion step. */
    BLI_assert(time_steps_.size() > step_id_ + 2);
    step_id_ += 2;

    if (motion_blur_fx_enabled_) {
      inst_.velocity.step_swap();
      inst_.velocity.step_sync(VelocityModule::STEP_NEXT, time_steps_[step_id_ + 1]);
    }
    inst_.set_time(time_steps_[step_id_]);
  }
}

float MotionBlurModule::shutter_time_to_scene_time(float time)
{
  switch (motion_blur_position_) {
    case SCE_EEVEE_MB_START:
      /* No offset. */
      break;
    case SCE_EEVEE_MB_CENTER:
      time -= 0.5f;
      break;
    case SCE_EEVEE_MB_END:
      time -= 1.0;
      break;
    default:
      BLI_assert(!"Invalid motion blur position enum!");
      break;
  }
  time *= motion_blur_shutter_;
  time += frame_time_;
  return time;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name MotionBlur
 *
 * \{ */

void MotionBlur::init()
{
  const Scene *scene = inst_.scene;
  data_.blur_max = scene->eevee.motion_blur_max;
  data_.depth_scale = scene->eevee.motion_blur_depth_scale;
  enabled_ = ((scene->eevee.flag & SCE_EEVEE_MOTION_BLUR_ENABLED) != 0) && (data_.blur_max > 0.5f);
}

void MotionBlur::sync(int extent[2])
{
  if (!enabled_) {
    return;
  }

  DrawEngineType *owner = (DrawEngineType *)view_name_.c_str();
  eGPUSamplerState no_filter = GPU_SAMPLER_DEFAULT;

  uint res[2] = {divide_ceil_u(extent[0], MB_TILE_DIVISOR),
                 divide_ceil_u(extent[1], MB_TILE_DIVISOR)};

  {
    /* Create max velocity tiles in 2 passes. One for X and one for Y */
    DRW_PASS_CREATE(tiles_flatten_ps_, DRW_STATE_WRITE_COLOR);
    GPUShader *sh = inst_.shaders.static_shader_get(MOTION_BLUR_TILE_FLATTEN);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, tiles_flatten_ps_);
    DRW_shgroup_uniform_texture_ref_ex(grp, "velocity_tx", &input_velocity_tx_, no_filter);
    DRW_shgroup_uniform_block(grp, "motion_blur_block", data_.ubo_get());
    DRW_shgroup_call_procedural_triangles(grp, NULL, 1);

    tiles_tx_ = DRW_texture_pool_query_2d(UNPACK2(res), GPU_RGBA16F, owner);

    tiles_flatten_fb_.ensure(GPU_ATTACHMENT_NONE, GPU_ATTACHMENT_TEXTURE(tiles_tx_));
  }
  {
    /* Expand max tiles by keeping the max tile in each tile neighborhood. */
    DRW_PASS_CREATE(tiles_dilate_ps_, DRW_STATE_WRITE_COLOR);
    GPUShader *sh = inst_.shaders.static_shader_get(MOTION_BLUR_TILE_DILATE);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, tiles_dilate_ps_);
    DRW_shgroup_uniform_texture_ref_ex(grp, "tiles_tx", &tiles_tx_, no_filter);
    DRW_shgroup_uniform_block(grp, "motion_blur_block", data_.ubo_get());
    DRW_shgroup_call_procedural_triangles(grp, NULL, 1);

    tiles_dilated_tx_ = DRW_texture_pool_query_2d(UNPACK2(res), GPU_RGBA16F, owner);

    tiles_dilate_fb_.ensure(GPU_ATTACHMENT_NONE, GPU_ATTACHMENT_TEXTURE(tiles_dilated_tx_));
  }
  {
    data_.target_size_inv[0] = 1.0f / extent[0];
    data_.target_size_inv[1] = 1.0f / extent[1];

    /* Do the motion blur gather algorithm. */
    DRW_PASS_CREATE(gather_ps_, DRW_STATE_WRITE_COLOR);
    GPUShader *sh = inst_.shaders.static_shader_get(MOTION_BLUR_GATHER);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, gather_ps_);
    DRW_shgroup_uniform_block(grp, "sampling_block", inst_.sampling.ubo_get());
    DRW_shgroup_uniform_block(grp, "motion_blur_block", data_.ubo_get());
    DRW_shgroup_uniform_texture_ref(grp, "color_tx", &input_color_tx_);
    DRW_shgroup_uniform_texture_ref(grp, "depth_tx", &input_depth_tx_);
    DRW_shgroup_uniform_texture_ref_ex(grp, "velocity_tx", &input_velocity_tx_, no_filter);
    DRW_shgroup_uniform_texture_ref_ex(grp, "tiles_tx", &tiles_dilated_tx_, no_filter);

    DRW_shgroup_call_procedural_triangles(grp, NULL, 1);
  }

  data_.is_viewport = !DRW_state_is_image_render();
  data_.push_update();
}

void MotionBlur::render(GPUTexture *depth_tx,
                        GPUTexture *velocity_tx,
                        GPUTexture **input_tx,
                        GPUTexture **output_tx)
{
  if (!enabled_) {
    return;
  }

  input_color_tx_ = *input_tx;
  input_depth_tx_ = depth_tx;
  input_velocity_tx_ = velocity_tx;

  DRW_stats_group_start("Motion Blur");

  GPU_framebuffer_bind(tiles_flatten_fb_);
  DRW_draw_pass(tiles_flatten_ps_);

  for (int max_blur = data_.blur_max; max_blur > 0; max_blur -= MB_TILE_DIVISOR) {
    GPU_framebuffer_bind(tiles_dilate_fb_);
    DRW_draw_pass(tiles_dilate_ps_);
    SWAP(GPUTexture *, tiles_tx_, tiles_dilated_tx_);
    SWAP(Framebuffer, tiles_flatten_fb_, tiles_dilate_fb_);
  }
  /* Swap again so result is in tiles_dilated_tx_. */
  SWAP(GPUTexture *, tiles_tx_, tiles_dilated_tx_);
  SWAP(Framebuffer, tiles_flatten_fb_, tiles_dilate_fb_);

  gather_fb_.ensure(GPU_ATTACHMENT_NONE, GPU_ATTACHMENT_TEXTURE(*output_tx));

  GPU_framebuffer_bind(gather_fb_);
  DRW_draw_pass(gather_ps_);

  DRW_stats_group_end();

  /* Swap buffers so that next effect has the right input. */
  *input_tx = *output_tx;
  *output_tx = input_color_tx_;
}

/** \} */

}  // namespace blender::eevee