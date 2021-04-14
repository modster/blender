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

#include "BKE_duplilist.h"
#include "BLI_map.hh"

#include "eevee_renderpasses.hh"
#include "eevee_shader_shared.hh"

namespace blender::eevee {

/* -------------------------------------------------------------------- */
/** \name ObjectKey
 *
 * Unique key to be able to match object across frame updates.
 * \{ */

/** Unique key to identify each object in the hashmap. */
struct ObjectKey {
  /** Original Object or source object for duplis. */
  Object *ob;
  /** Original Parent object for duplis. */
  Object *parent;
  /** Dupli objects recursive unique identifier */
  int id[8]; /* MAX_DUPLI_RECUR */
  /** If object uses particle system hair. */
  bool use_particle_hair;

  ObjectKey(Object *ob_,
            Object *parent_,
            int id_[8], /* MAX_DUPLI_RECUR */
            bool use_particle_hair_)
      : ob(ob_), parent(parent_), use_particle_hair(use_particle_hair_)
  {
    if (id_) {
      memcpy(id, id_, sizeof(id));
    }
    else {
      memset(id, 0, sizeof(id));
    }
  }

  ObjectKey(Object *ob, DupliObject *dupli, Object *parent)
      : ObjectKey(ob, parent, dupli ? dupli->persistent_id : nullptr, false){};

  ObjectKey(Object *ob)
      : ObjectKey(ob, DRW_object_get_dupli(ob), DRW_object_get_dupli_parent(ob)){};

  uint64_t hash(void) const
  {
    uint64_t hash = BLI_ghashutil_ptrhash(ob);
    hash = BLI_ghashutil_combine_hash(hash, BLI_ghashutil_ptrhash(parent));
    for (int i = 0; i < MAX_DUPLI_RECUR; i++) {
      if (id[i] != 0) {
        hash = BLI_ghashutil_combine_hash(hash, BLI_ghashutil_inthash(id[i]));
      }
      else {
        break;
      }
    }
    return hash;
  }

  bool operator<(const ObjectKey &k) const
  {
    if (ob != k.ob) {
      return (ob < k.ob);
    }
    if (parent != k.parent) {
      return (parent < k.parent);
    }
    if (use_particle_hair != k.use_particle_hair) {
      return (use_particle_hair < k.use_particle_hair);
    }
    return memcmp(id, k.id, sizeof(id)) < 0;
  }

  bool operator==(const ObjectKey &k) const
  {
    if (ob != k.ob) {
      return false;
    }
    if (parent != k.parent) {
      return false;
    }
    if (use_particle_hair != k.use_particle_hair) {
      return false;
    }
    return memcmp(id, k.id, sizeof(id)) == 0;
  }
};

/** \} */

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

  /** True if velocity is computed for viewport. */
  bool is_viewport_;

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
  static void step_object_sync(void *velocity,
                               Object *ob,
                               RenderEngine *UNUSED(engine),
                               Depsgraph *UNUSED(depsgraph));

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

  void mesh_add(Object *ob);

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
