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
 * \{ */

/**
 * Manages timesteps evaluations and accumulation Motion blur.
 * Post process motion blur is handled by the MotionBlur class.
 */
class MotionBlurModule {
 private:
  Instance &inst_;

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

  bool enabled_ = false;
  float motion_blur_fx_enabled_ = false;

  int step_id_ = 0;

 public:
  MotionBlurModule(Instance &inst) : inst_(inst){};
  ~MotionBlurModule(){};

  void init(void);

  void step(void);

 private:
  float shutter_time_to_scene_time(float time);
};

/** \} */

/* -------------------------------------------------------------------- */
/** \name MotionBlur
 *
 * \{ */

/**
 * Per view fx module. Perform a motion blur using the result of the velocity pass.
 */
class MotionBlur {
 private:
  Instance &inst_;

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
  eevee::Framebuffer tiles_flatten_fb_;
  eevee::Framebuffer tiles_dilate_fb_;
  eevee::Framebuffer gather_fb_;

  StructBuffer<MotionBlurData> data_;

  bool enabled_;

 public:
  MotionBlur(Instance &inst, StringRefNull view_name) : inst_(inst), view_name_(view_name){};

  ~MotionBlur(){};

  void init(void);

  void sync(int extent[2]);

  void render(GPUTexture *depth_tx,
              GPUTexture *velocity_tx,
              GPUTexture **input_tx,
              GPUTexture **output_tx);
};

/** \} */

}  // namespace blender::eevee
