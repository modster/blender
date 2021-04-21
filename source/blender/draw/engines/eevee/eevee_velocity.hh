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
 * The velocity pass outputs motion vectors to use for either
 * temporal re-projection or motion blur.
 *
 * It is the module that tracks the objects between frames updates.
 */

#pragma once

#include "BLI_map.hh"

#include "eevee_id_map.hh"
#include "eevee_renderpasses.hh"
#include "eevee_shader_shared.hh"

namespace blender::eevee {

/* -------------------------------------------------------------------- */
/** \name VelocityModule
 *
 * \{ */

/** Container for scene velocity data. */
class VelocityModule {
 public:
  enum eStep {
    STEP_PREVIOUS = 0,
    STEP_NEXT = 1,
    STEP_CURRENT = 2,
  };

  /** Map an object key to a velocity data. */
  Map<ObjectKey, VelocityObjectBuf *> objects_steps;
  struct {
    /** Copies of camera data. One for previous and one for next time step. */
    StructBuffer<CameraData> prev, next;
  } camera_step;

 private:
  Instance &inst_;

  eStep step_;

 public:
  VelocityModule(Instance &inst) : inst_(inst){};

  ~VelocityModule()
  {
    for (VelocityObjectBuf *data : objects_steps.values()) {
      delete data;
    }
  }

  void init(void);

  void step_camera_sync(void);
  void step_sync(eStep step, float time);

  /* Gather motion data from all objects in the scene. */
  void step_object_sync(Object *ob, ObjectKey &ob_key);

  /* Moves next frame data to previous frame data. Nullify next frame data. */
  void step_swap(void);

  void begin_sync(void);
  void end_sync(void);

 private:
  bool object_has_velocity(const Object *ob);
  bool object_is_deform(const Object *ob);
};

/** \} */

/* -------------------------------------------------------------------- */
/** \name VelocityPass
 *
 * Draws velocity data from VelocityModule module to a framebuffer / texture.
 * \{ */

class VelocityPass {
 private:
  Instance &inst_;

  DRWPass *object_ps_ = nullptr;
  DRWPass *camera_ps_ = nullptr;

  /** Shading groups from object_ps_ */
  DRWShadingGroup *mesh_grp_;

  /** Reference only. Not owned. */
  GPUTexture *input_depth_tx_;

 public:
  VelocityPass(Instance &inst) : inst_(inst){};

  void sync(void);

  void mesh_add(Object *ob, ObjectHandle &handle);

  void render_objects(void);
  void resolve_camera_motion(GPUTexture *depth_tx);
};

/** \} */

/* -------------------------------------------------------------------- */
/** \name Velocity
 *
 * \{ */

/**
 * Per view module.
 */
class Velocity {
 private:
  Instance &inst_;

  StringRefNull view_name_;

  /** Owned resources. */
  eevee::Framebuffer velocity_fb_;
  eevee::Framebuffer velocity_only_fb_;
  /** Draw resources. Not owned. */
  GPUTexture *velocity_camera_tx_ = nullptr;
  GPUTexture *velocity_view_tx_ = nullptr;

 public:
  Velocity(Instance &inst, const char *name) : inst_(inst), view_name_(name){};
  ~Velocity(){};

  void sync(int extent[2]);

  void render(GPUTexture *depth_tx);

  /**
   * Getters
   **/
  GPUTexture *view_vectors_get(void) const
  {
    return (velocity_view_tx_ != nullptr) ? velocity_view_tx_ : velocity_camera_tx_;
  }
  GPUTexture *camera_vectors_get(void) const
  {
    return velocity_camera_tx_;
  }
};

/** \} */

}  // namespace blender::eevee
