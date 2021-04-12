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
 * Motion blur is done by accumulating scene samples over shutter time.
 * Since the number of step is discrete, quite low, and not per pixel randomized,
 * we couple this with a post processing motion blur.
 *
 * The post-fx motion blur is done in two directions, from the previous step and to the next.
 *
 * For a scene with 3 motion steps, a flat shutter curve and shutter time of 2 frame
 * centered on frame we have:
 *
 * |--------------------|--------------------|
 * -1                   0                    1  Frames
 *
 * |-------------|-------------|-------------|
 *        1             2             3         Motion steps
 *
 * |------|------|------|------|------|------|
 * 0      1      2      4      5      6      7  Time Steps
 *
 * |-------------| One motion step blurs this range.
 * -1     |     +1 Objects and geometry steps are recorded here.
 *        0 Scene is rendered here.
 *
 * Since motion step N and N+1 share one time step we reuse it to avoid an extra scene evaluation.
 *
 * Note that we have to evaluate -1 and +1 time steps before rendering so eval order is -1, +1, 0.
 * This is because all GPUBatches from the DRWCache are being free when changing a frame.
 */

#pragma once

#include "BLI_map.hh"
#include "DEG_depsgraph_query.h"

#include "eevee_sampling.hh"
#include "eevee_shader_shared.hh"
#include "eevee_velocity.hh"

namespace blender::eevee {

/* -------------------------------------------------------------------- */
/** \name MotionBlur
 *
 * Common module. Manages timesteps evaluations and accumulation Motion blur.
 * \{ */

class MotionBlurModule {
 private:
  Sampling &sampling_;
  Velocity &velocity_;
  Camera &camera_;

  RenderEngine *engine_;
  Depsgraph *depsgraph_;

  /**
   * Array containing all steps (in scene time) we need to evaluate (not render).
   * Only odd steps are rendered. The even ones are evaluated for fx motion blur.
   */
  Vector<float> time_steps_;

  /** Copy of input frame an subframe to restore after render. */
  int initial_frame_;
  float initial_subframe_;
  /** Time of the frame we are rendering. */
  float frame_time_;
  /** Copy of scene settings. */
  int motion_blur_position_;
  float motion_blur_shutter_;
  float motion_blur_fx_depth_scale_;

  bool enabled_ = false;
  float motion_blur_fx_enabled_ = false;

  int step_id_ = 0;

 public:
  MotionBlurModule(Camera &camera, Sampling &sampling, Velocity &velocity)
      : sampling_(sampling), velocity_(velocity), camera_(camera){};
  ~MotionBlurModule(){};

  void init(const Scene *scene, RenderEngine *engine, Depsgraph *depsgraph)
  {
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

    engine_ = engine;
    depsgraph_ = depsgraph;

    if (motion_blur_fx_enabled_) {
      /* A bit weird but we have to sync the first 2 steps here because the step()
       * function is only called rendering a sample. */
      velocity_.step_sync(Velocity::STEP_PREVIOUS, camera_, engine_, depsgraph_, time_steps_[0]);
      velocity_.step_sync(Velocity::STEP_NEXT, camera_, engine_, depsgraph_, time_steps_[2]);
    }
    float frame_time = time_steps_[1];
    DRW_render_set_time(engine_, depsgraph_, floorf(frame_time), fractf(frame_time));
  }

  /* Runs after rendering a sample. */
  void step(void)
  {
    if (!enabled_) {
      return;
    }
    else if (sampling_.finished()) {
      /* Restore original frame number. This is because the render pipeline expects it. */
      RE_engine_frame_set(engine_, initial_frame_, initial_subframe_);
    }
    else if (sampling_.do_render_sync()) {
      /* Time to change motion step. */
      BLI_assert(time_steps_.size() > step_id_ + 2);
      step_id_ += 2;

      if (motion_blur_fx_enabled_) {
        velocity_.step_swap();
        velocity_.step_sync(
            Velocity::STEP_NEXT, camera_, engine_, depsgraph_, time_steps_[step_id_ + 1]);
      }
      float frame_time = time_steps_[step_id_];
      DRW_render_set_time(engine_, depsgraph_, floorf(frame_time), fractf(frame_time));
    }
  }

  bool blur_fx_enabled_get(void) const
  {
    return motion_blur_fx_enabled_;
  }

 private:
  float shutter_time_to_scene_time(float time)
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
};

/** \} */

/* -------------------------------------------------------------------- */
/** \name MotionBlur
 *
 * Per view fx module. Perform a motion blur using the result of the velocity pass.
 * \{ */

class MotionBlur {
 private:
  ShaderModule &shaders_;
  Sampling &sampling_;
  MotionBlurModule &mb_module_;

  StringRefNull view_name_;

  /** Textures from pool. Not owned. */
  GPUTexture *tiles_tx_ = nullptr;
  GPUTexture *tiles_dilated_tx_ = nullptr;
  /** Input texture. Not owned. */
  GPUTexture *input_velocity_tx_ = nullptr;
  GPUTexture *input_color_tx_ = nullptr;
  GPUTexture *input_depth_tx_ = nullptr;
  /** Passes. Not owned. */
  DRWPass *tiles_flatten_ps_ = nullptr;
  DRWPass *tiles_dilate_ps_ = nullptr;
  DRWPass *gather_ps_ = nullptr;
  /** Framebuffers. Owned.  */
  GPUFrameBuffer *tiles_flatten_fb_ = nullptr;
  GPUFrameBuffer *tiles_dilate_fb_ = nullptr;
  GPUFrameBuffer *gather_fb_ = nullptr;

  StructBuffer<MotionBlurData> data_;

  bool enabled_;

 public:
  MotionBlur(ShaderModule &shaders,
             Sampling &sampling,
             MotionBlurModule &mb_module,
             StringRefNull view_name)
      : shaders_(shaders), sampling_(sampling), mb_module_(mb_module), view_name_(view_name){};

  ~MotionBlur()
  {
    GPU_FRAMEBUFFER_FREE_SAFE(tiles_flatten_fb_);
    GPU_FRAMEBUFFER_FREE_SAFE(tiles_dilate_fb_);
    GPU_FRAMEBUFFER_FREE_SAFE(gather_fb_);
  }

  void init(const Scene *scene)
  {
    data_.blur_max = scene->eevee.motion_blur_max;
    data_.depth_scale = scene->eevee.motion_blur_depth_scale;
    enabled_ = ((scene->eevee.flag & SCE_EEVEE_MOTION_BLUR_ENABLED) != 0) &&
               (data_.blur_max > 0.5f);
  }

  void sync(int extent[2])
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
      GPUShader *sh = shaders_.static_shader_get(MOTION_BLUR_TILE_FLATTEN);
      DRWShadingGroup *grp = DRW_shgroup_create(sh, tiles_flatten_ps_);
      DRW_shgroup_uniform_texture_ref_ex(grp, "velocity_tx", &input_velocity_tx_, no_filter);
      DRW_shgroup_uniform_block(grp, "motion_blur_block", data_.ubo_get());
      DRW_shgroup_call_procedural_triangles(grp, NULL, 1);

      tiles_tx_ = DRW_texture_pool_query_2d(UNPACK2(res), GPU_RGBA16F, owner);

      GPU_framebuffer_ensure_config(&tiles_flatten_fb_,
                                    {
                                        GPU_ATTACHMENT_NONE,
                                        GPU_ATTACHMENT_TEXTURE(tiles_tx_),
                                    });
    }
    {
      /* Expand max tiles by keeping the max tile in each tile neighborhood. */
      DRW_PASS_CREATE(tiles_dilate_ps_, DRW_STATE_WRITE_COLOR);
      GPUShader *sh = shaders_.static_shader_get(MOTION_BLUR_TILE_DILATE);
      DRWShadingGroup *grp = DRW_shgroup_create(sh, tiles_dilate_ps_);
      DRW_shgroup_uniform_texture_ref_ex(grp, "tiles_tx", &tiles_tx_, no_filter);
      DRW_shgroup_uniform_block(grp, "motion_blur_block", data_.ubo_get());
      DRW_shgroup_call_procedural_triangles(grp, NULL, 1);

      tiles_dilated_tx_ = DRW_texture_pool_query_2d(UNPACK2(res), GPU_RGBA16F, owner);

      GPU_framebuffer_ensure_config(&tiles_dilate_fb_,
                                    {
                                        GPU_ATTACHMENT_NONE,
                                        GPU_ATTACHMENT_TEXTURE(tiles_dilated_tx_),
                                    });
    }
    {
      data_.target_size_inv[0] = 1.0f / extent[0];
      data_.target_size_inv[1] = 1.0f / extent[1];

      /* Do the motion blur gather algorithm. */
      DRW_PASS_CREATE(gather_ps_, DRW_STATE_WRITE_COLOR);
      GPUShader *sh = shaders_.static_shader_get(MOTION_BLUR_GATHER);
      DRWShadingGroup *grp = DRW_shgroup_create(sh, gather_ps_);
      DRW_shgroup_uniform_block(grp, "sampling_block", sampling_.ubo_get());
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

  void render(GPUTexture *depth_tx,
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
      SWAP(GPUFrameBuffer *, tiles_flatten_fb_, tiles_dilate_fb_);
    }
    /* Swap again so result is in tiles_dilated_tx_. */
    SWAP(GPUTexture *, tiles_tx_, tiles_dilated_tx_);
    SWAP(GPUFrameBuffer *, tiles_flatten_fb_, tiles_dilate_fb_);

    GPU_framebuffer_ensure_config(&gather_fb_,
                                  {
                                      GPU_ATTACHMENT_NONE,
                                      GPU_ATTACHMENT_TEXTURE(*output_tx),
                                  });

    GPU_framebuffer_bind(gather_fb_);
    DRW_draw_pass(gather_ps_);

    DRW_stats_group_end();

    /* Swap buffers so that next effect has the right input. */
    *input_tx = *output_tx;
    *output_tx = input_color_tx_;
  }
};

/** \} */

}  // namespace blender::eevee
