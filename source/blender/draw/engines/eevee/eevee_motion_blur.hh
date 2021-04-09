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

namespace blender::eevee {

class MotionBlur {
 public:
  struct ViewStep {
    CameraData cam_data;
  };

 private:
  enum eStep {
    STEP_PREVIOUS = 0,
    STEP_NEXT = 1,
    STEP_CURRENT = 2,
  };

  struct ObjectSteps {
    mat4 obmats[3];
  };

  struct HairSteps {
    /** Boolean used as uniform to disable deformation motion-blur just before drawing. */
    int use_deform;
    /** Position buffer for time = t +/- step. */
    struct GPUVertBuf *hair_pos[2] = {nullptr};
    /** Buffer Texture of the corresponding VBO. */
    struct GPUTexture *hair_pos_tx[2] = {nullptr};
  };

  struct GeometrySteps : public ObjectSteps {
    /** Boolean used as uniform to disable deformation motion-blur just before drawing. */
    int use_deform;
    /** Batch for time = t. */
    struct GPUBatch *batch = nullptr;
    /** Vbo for time = t +/- step. */
    struct GPUVertBuf *vbo[2] = {nullptr};
    /** Hair motion steps for particle systems. */
    Vector<HairSteps> psys;
  };

  /** Unique key to identify each object in the hashmap. */
  struct ObjectKey {
    /** Original Object or source object for duplis. */
    struct Object *ob = nullptr;
    /** Original Parent object for duplis. */
    struct Object *parent = nullptr;
    /** Dupli objects recursive unique identifier */
    int id[8] = {0}; /* MAX_DUPLI_RECUR */
  };

  ViewStep camera_steps[3];
  Map<ObjectKey, GeometrySteps *> geom_steps_;
  Map<ObjectKey, HairSteps *> hair_steps_;

  Sampling &sampling_;

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

  /**  */
  bool use_fx_motion_blur = false;
  bool enabled_ = false;

  eStep step_type_ = STEP_CURRENT;
  int step_id_ = 0;

 public:
  MotionBlur(Sampling &sampling) : sampling_(sampling){};
  ~MotionBlur(){};

  void init(const Scene *scene, RenderEngine *engine, Depsgraph *depsgraph)
  {
    enabled_ = (scene->eevee.flag & SCE_EEVEE_MOTION_BLUR_ENABLED) != 0;

    /* Viewport not supported for now. */
    if (!DRW_state_is_scene_render()) {
      enabled_ = false;
    }
    if (!enabled_) {
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

    // use_fx_motion_blur = scene->eevee.motion_blur_max > 0.0f;
    use_fx_motion_blur = false;
    step_id_ = 1;

    engine_ = engine;
    depsgraph_ = depsgraph;

    if (use_fx_motion_blur) {
      /* A bit weird but we have to sync the first 2 steps here because the step()
       * function is only called. */
      this->step_sync(time_steps_[0], STEP_PREVIOUS);
      this->step_sync(time_steps_[2], STEP_NEXT);
    }
    this->set_time(time_steps_[1]);
  }

  /* Runs once per center time step. */
  void end_sync(void)
  {
    if (!enabled_) {
      return;
    }
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

      if (use_fx_motion_blur) {
        this->step_swap();
        this->step_sync(time_steps_[step_id_ + 1], STEP_NEXT);
      }
      step_type_ = STEP_CURRENT;
      this->set_time(time_steps_[step_id_]);
    }
  }

 private:
  /* Gather motion data from all objects in the scene. */
  static void step_object_sync(void *motion_blur_,
                               Object *ob,
                               RenderEngine *UNUSED(engine),
                               Depsgraph *UNUSED(depsgraph))
  {
    MotionBlur &mb = *reinterpret_cast<MotionBlur *>(motion_blur_);
    (void)mb;

    switch (ob->type) {
      case OB_MESH:
        break;

      default:
        break;
    }
  }

  void step_sync(float time, eStep step)
  {
    step_type_ = step;
    this->set_time(time);
    DRW_render_object_iter(this, engine_, depsgraph_, MotionBlur::step_object_sync);
  }

  /* Swaps next frame data  */
  void step_swap()
  {
  }

  void set_time(float time)
  {
    DRW_render_set_time(engine_, depsgraph_, floorf(time), fractf(time));
  }

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

}  // namespace blender::eevee
