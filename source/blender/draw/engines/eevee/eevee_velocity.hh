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
#include "BKE_object.h"
#include "BLI_map.hh"
#include "DEG_depsgraph_query.h"
#include "DNA_rigidbody_types.h"
#include "GPU_framebuffer.h"

#include "eevee_renderpasses.hh"
#include "eevee_shader.hh"
#include "eevee_shader_shared.hh"
#include "eevee_wrapper.hh"

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
/** \name Velocity
 *
 * Container for scene velocity data.
 * \{ */

using VelocityObjectBuf = StructBuffer<VelocityObjectData>;

class Velocity {
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
  Scene *scene_;
  eStep step_;

  /** True if velocity is computed for viewport. */
  bool is_viewport_;

 public:
  Velocity(){};

  ~Velocity()
  {
    for (VelocityObjectBuf *data : objects_steps.values()) {
      delete data;
    }
  }

  void init(Camera &camera,
            RenderEngine *engine,
            Depsgraph *depsgraph,
            const RenderPasses &rpasses)
  {
    is_viewport_ = !DRW_state_is_image_render() && !DRW_state_is_opengl_render();

    if (is_viewport_) {
      /* For viewport we sync when object is evaluated and we swap at init time.
       * Use next step to store the current position. This one will become the previous step after
       * next swapping. */
      step_ = STEP_NEXT;
      step_swap();
      /* TODO(fclem) we should garbage collect the ids that gets removed. */
    }

    if (engine && (rpasses.vector != nullptr)) {
      /* No motion blur and the vector pass was requested. Do the step sync here. */
      Scene *scene = DEG_get_evaluated_scene(depsgraph);
      float initial_time = scene->r.cfra + scene->r.subframe;
      step_sync(STEP_PREVIOUS, camera, engine, depsgraph, initial_time - 1.0f);
      step_sync(STEP_NEXT, camera, engine, depsgraph, initial_time + 1.0f);
      DRW_render_set_time(engine, depsgraph, floorf(initial_time), fractf(initial_time));
    }
  }

  void step_sync(
      eStep step, Camera &camera, RenderEngine *engine, Depsgraph *depsgraph, float time)
  {
    DRW_render_set_time(engine, depsgraph, floorf(time), fractf(time));
    step_ = step;
    scene_ = DEG_get_evaluated_scene(depsgraph);
    step_camera_sync(camera);
    DRW_render_object_iter(this, engine, depsgraph, Velocity::step_object_sync);
  }

  void step_camera_sync(Camera &camera)
  {
    if (!is_viewport_) {
      camera.sync();
    }

    if (step_ == STEP_NEXT) {
      camera_step.next = camera.data_get();
    }
    else if (step_ == STEP_PREVIOUS) {
      camera_step.prev = camera.data_get();
    }
  }

  /* Gather motion data from all objects in the scene. */
  static void step_object_sync(void *velocity_,
                               Object *ob,
                               RenderEngine *UNUSED(engine),
                               Depsgraph *UNUSED(depsgraph))
  {
    Velocity &velocity = *reinterpret_cast<Velocity *>(velocity_);

    if (!velocity.object_has_velocity(ob) && !velocity.object_is_deform(ob)) {
      return;
    }

    auto data = velocity.objects_steps.lookup_or_add_cb(ObjectKey(ob),
                                                        []() { return new VelocityObjectBuf(); });

    if (velocity.step_ == STEP_NEXT) {
      copy_m4_m4(data->next_object_mat, ob->obmat);
    }
    else if (velocity.step_ == STEP_PREVIOUS) {
      copy_m4_m4(data->prev_object_mat, ob->obmat);
    }
  }

  /* Moves next frame data to previous frame data. Nullify next frame data. */
  void step_swap(void)
  {
    for (VelocityObjectBuf *data : objects_steps.values()) {
      copy_m4_m4(data->prev_object_mat, data->next_object_mat);
      /* Important: This let us known if object is missing from the next time step. */
      zero_m4(data->next_object_mat);
    }
    camera_step.prev = static_cast<CameraData>(camera_step.next);
  }

  void begin_sync(Camera &camera)
  {
    if (is_viewport_) {
      step_camera_sync(camera);
    }
  }

  /* This is the end of the current frame sync. Not the step_sync. */
  void end_sync(void)
  {
    for (VelocityObjectBuf *data : objects_steps.values()) {
      data->push_update();
    }
    camera_step.prev.push_update();
    camera_step.next.push_update();
  }

 private:
  bool object_has_velocity(const Object *ob)
  {
#if 0
    RigidBodyOb *rbo = ob->rigidbody_object;
    /* Active rigidbody objects only, as only those are affected by sim. */
    const bool has_rigidbody = (rbo && (rbo->type == RBO_TYPE_ACTIVE));
    /* For now we assume dupli objects are moving. */
    const bool is_dupli = (ob->base_flag & BASE_FROM_DUPLI) != 0;
    const bool object_moves = is_dupli || has_rigidbody || BKE_object_moves_in_time(ob, true);
#else
    UNUSED_VARS(ob);
    /* BKE_object_moves_in_time does not work in some cases.
     * Better detect non moving object after evaluation. */
    const bool object_moves = true;
#endif
    return object_moves;
  }

  bool object_is_deform(const Object *ob)
  {
    RigidBodyOb *rbo = ob->rigidbody_object;
    /* Active rigidbody objects only, as only those are affected by sim. */
    const bool has_rigidbody = (rbo && (rbo->type == RBO_TYPE_ACTIVE));
    const bool is_deform = BKE_object_is_deform_modified(scene_, (Object *)ob) ||
                           (has_rigidbody && (rbo->flag & RBO_FLAG_USE_DEFORM) != 0);

    return is_deform;
  }
};

/** \} */

/* -------------------------------------------------------------------- */
/** \name VelocityPass
 *
 * Draws velocity data from Velocity module to a framebuffer / texture.
 * \{ */

class VelocityPass {
 private:
  ShaderModule &shaders_;
  Camera &camera_;
  Velocity &velocity_;

  DRWPass *object_ps_ = nullptr;
  DRWPass *camera_ps_ = nullptr;

  /** Shading groups from object_ps_ */
  DRWShadingGroup *mesh_grp_;

  /** Reference only. Not owned. */
  GPUTexture *depth_tx_;

 public:
  VelocityPass(ShaderModule &shaders, Camera &camera, Velocity &velocity)
      : shaders_(shaders), camera_(camera), velocity_(velocity){};

  void sync(void)
  {
    {
      /* Outputs camera motion vector. */
      /* TODO(fclem) Ideally, we should run this only where the motion vectors were not written.
       * But without imageLoadStore, we cannot do that without another buffer. */
      DRWState state = DRW_STATE_WRITE_COLOR;
      DRW_PASS_CREATE(camera_ps_, state);
      GPUShader *sh = shaders_.static_shader_get(VELOCITY_CAMERA);
      DRWShadingGroup *grp = DRW_shgroup_create(sh, camera_ps_);
      DRW_shgroup_uniform_texture_ref(grp, "depth_tx", &depth_tx_);
      DRW_shgroup_uniform_block(grp, "camera_prev_block", velocity_.camera_step.prev.ubo_get());
      DRW_shgroup_uniform_block(grp, "camera_next_block", velocity_.camera_step.next.ubo_get());
      DRW_shgroup_uniform_block(grp, "camera_curr_block", camera_.ubo_get());
      DRW_shgroup_call_procedural_triangles(grp, nullptr, 1);
    }
    {
      /* Animated objects are rendered and output the correct motion vector. */
      DRWState state = DRW_STATE_WRITE_COLOR | DRW_STATE_DEPTH_EQUAL;
      DRW_PASS_CREATE(object_ps_, state);
      {
        GPUShader *sh = shaders_.static_shader_get(VELOCITY_MESH);
        DRWShadingGroup *grp = mesh_grp_ = DRW_shgroup_create(sh, object_ps_);
        DRW_shgroup_uniform_block(grp, "camera_prev_block", velocity_.camera_step.prev.ubo_get());
        DRW_shgroup_uniform_block(grp, "camera_next_block", velocity_.camera_step.next.ubo_get());
        DRW_shgroup_uniform_block(grp, "camera_curr_block", camera_.ubo_get());
      }
    }
  }

  void mesh_add(Object *ob)
  {
    VelocityObjectBuf **data_ptr = velocity_.objects_steps.lookup_ptr(ObjectKey(ob));

    if (data_ptr == nullptr) {
      return;
    }

    VelocityObjectBuf *data = *data_ptr;

    GPUBatch *geom = DRW_cache_object_surface_get(ob);
    if (geom == NULL) {
      return;
    }

    /* Fill missing matrices if the object was hidden in previous or next frame. */
    if (is_zero_m4(data->prev_object_mat)) {
      copy_m4_m4(data->prev_object_mat, ob->obmat);
    }
    if (is_zero_m4(data->next_object_mat)) {
      copy_m4_m4(data->next_object_mat, ob->obmat);
    }

    // if (mb_geom->use_deform) {
    //   /* Keep to modify later (after init). */
    //   mb_geom->batch = geom;
    // }

    /* Avoid drawing object that has no motions since object_moves is always true. */
    if (/* !mb_geom->use_deform && */ /* Object deformation can happen without transform.  */
        equals_m4m4(data->prev_object_mat, ob->obmat) &&
        equals_m4m4(data->next_object_mat, ob->obmat)) {
      return;
    }

    /* TODO(fclem) Use the same layout as modelBlock from draw so we can reuse the same offset and
     * avoid the overhead of 1 shading group and one UBO per object. */
    DRWShadingGroup *grp = DRW_shgroup_create_sub(mesh_grp_);
    DRW_shgroup_uniform_block(grp, "object_block", data->ubo_get());
    DRW_shgroup_call(grp, geom, ob);
  }

  void render(GPUTexture *depth_tx, GPUFrameBuffer *velocity_only_fb, GPUFrameBuffer *velocity_fb)
  {
    depth_tx_ = depth_tx;

    DRW_stats_group_start("Velocity");

    GPU_framebuffer_bind(velocity_only_fb);
    DRW_draw_pass(camera_ps_);

    GPU_framebuffer_bind(velocity_fb);
    DRW_draw_pass(object_ps_);

    DRW_stats_group_end();
  }
};

/** \} */

}  // namespace blender::eevee
