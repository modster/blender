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

#include "BKE_duplilist.h"
#include "BKE_object.h"
#include "BLI_map.hh"
#include "DEG_depsgraph_query.h"
#include "DNA_rigidbody_types.h"

#include "eevee_instance.hh"
#include "eevee_renderpasses.hh"
#include "eevee_shader.hh"
#include "eevee_shader_shared.hh"
#include "eevee_velocity.hh"
#include "eevee_wrapper.hh"

namespace blender::eevee {

/* -------------------------------------------------------------------- */
/** \name VelocityModule
 *
 * \{ */

void VelocityModule::init(void)
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

  if (inst_.render && (inst_.render_passes.vector != nullptr)) {
    /* No motion blur and the vector pass was requested. Do the step sync here. */
    const Scene *scene = inst_.scene;
    float initial_time = scene->r.cfra + scene->r.subframe;
    step_sync(STEP_PREVIOUS, initial_time - 1.0f);
    step_sync(STEP_NEXT, initial_time + 1.0f);
    inst_.set_time(initial_time);
  }
}

void VelocityModule::step_sync(eStep step, float time)
{
  inst_.set_time(time);
  step_ = step;
  step_camera_sync();
  DRW_render_object_iter(this, inst_.render, inst_.depsgraph, VelocityModule::step_object_sync);
}

void VelocityModule::step_camera_sync()
{
  if (!is_viewport_) {
    inst_.camera.sync();
  }

  if (step_ == STEP_NEXT) {
    camera_step.next = inst_.camera.data_get();
  }
  else if (step_ == STEP_PREVIOUS) {
    camera_step.prev = inst_.camera.data_get();
  }
}

/* Gather motion data from all objects in the scene. */
void VelocityModule::step_object_sync(void *velocity_,
                                      Object *ob,
                                      RenderEngine *UNUSED(engine),
                                      Depsgraph *UNUSED(depsgraph))
{
  VelocityModule &velocity = *reinterpret_cast<VelocityModule *>(velocity_);

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
void VelocityModule::step_swap(void)
{
  for (VelocityObjectBuf *data : objects_steps.values()) {
    copy_m4_m4(data->prev_object_mat, data->next_object_mat);
    /* Important: This let us known if object is missing from the next time step. */
    zero_m4(data->next_object_mat);
  }
  camera_step.prev = static_cast<CameraData>(camera_step.next);
}

void VelocityModule::begin_sync(void)
{
  if (is_viewport_) {
    step_camera_sync();
  }
}

/* This is the end of the current frame sync. Not the step_sync. */
void VelocityModule::end_sync(void)
{
  for (VelocityObjectBuf *data : objects_steps.values()) {
    data->push_update();
  }
  camera_step.prev.push_update();
  camera_step.next.push_update();
}

bool VelocityModule::object_has_velocity(const Object *ob)
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

bool VelocityModule::object_is_deform(const Object *ob)
{
  RigidBodyOb *rbo = ob->rigidbody_object;
  /* Active rigidbody objects only, as only those are affected by sim. */
  const bool has_rigidbody = (rbo && (rbo->type == RBO_TYPE_ACTIVE));
  const bool is_deform = BKE_object_is_deform_modified(inst_.scene, (Object *)ob) ||
                         (has_rigidbody && (rbo->flag & RBO_FLAG_USE_DEFORM) != 0);

  return is_deform;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name VelocityPass
 *
 * Draws velocity data from VelocityModule module to a framebuffer / texture.
 * \{ */

void VelocityPass::sync(void)
{
  VelocityModule &velocity = inst_.velocity;
  {
    /* Outputs camera motion vector. */
    /* TODO(fclem) Ideally, we should run this only where the motion vectors were not written.
     * But without imageLoadStore, we cannot do that without another buffer. */
    DRWState state = DRW_STATE_WRITE_COLOR;
    DRW_PASS_CREATE(camera_ps_, state);
    GPUShader *sh = inst_.shaders.static_shader_get(VELOCITY_CAMERA);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, camera_ps_);
    DRW_shgroup_uniform_texture_ref(grp, "depth_tx", &input_depth_tx_);
    DRW_shgroup_uniform_block(grp, "camera_prev_block", velocity.camera_step.prev.ubo_get());
    DRW_shgroup_uniform_block(grp, "camera_next_block", velocity.camera_step.next.ubo_get());
    DRW_shgroup_uniform_block(grp, "camera_curr_block", inst_.camera.ubo_get());
    DRW_shgroup_call_procedural_triangles(grp, nullptr, 1);
  }
  {
    /* Animated objects are rendered and output the correct motion vector. */
    DRWState state = DRW_STATE_WRITE_COLOR | DRW_STATE_DEPTH_EQUAL;
    DRW_PASS_CREATE(object_ps_, state);
    {
      GPUShader *sh = inst_.shaders.static_shader_get(VELOCITY_MESH);
      DRWShadingGroup *grp = mesh_grp_ = DRW_shgroup_create(sh, object_ps_);
      DRW_shgroup_uniform_block(grp, "camera_prev_block", velocity.camera_step.prev.ubo_get());
      DRW_shgroup_uniform_block(grp, "camera_next_block", velocity.camera_step.next.ubo_get());
      DRW_shgroup_uniform_block(grp, "camera_curr_block", inst_.camera.ubo_get());
    }
  }
}

void VelocityPass::mesh_add(Object *ob)
{
  VelocityObjectBuf **data_ptr = inst_.velocity.objects_steps.lookup_ptr(ObjectKey(ob));

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

  /* TODO(fclem) Use the same layout as modelBlock from draw so we can reuse the same offset
   * and avoid the overhead of 1 shading group and one UBO per object. */
  DRWShadingGroup *grp = DRW_shgroup_create_sub(mesh_grp_);
  DRW_shgroup_uniform_block(grp, "object_block", data->ubo_get());
  DRW_shgroup_call(grp, geom, ob);
}

void VelocityPass::render_objects(void)
{
  DRW_draw_pass(object_ps_);
}

void VelocityPass::resolve_camera_motion(GPUTexture *depth_tx)
{
  input_depth_tx_ = depth_tx;
  DRW_draw_pass(camera_ps_);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name VelocityPass
 *
 * Draws velocity data from VelocityModule module to a framebuffer / texture.
 * \{ */

void Velocity::sync(int extent[2])
{
  /* HACK: View name should be unique and static.
   * With this, we can reuse the same texture across views. */
  DrawEngineType *owner = (DrawEngineType *)view_name_.c_str();

  /* TODO(fclem) Only allocate if needed. RG16F when only doing reprojection. */
  velocity_camera_tx_ = DRW_texture_pool_query_2d(UNPACK2(extent), GPU_RGBA16F, owner);
  /* TODO(fclem) Only allocate if needed. RG16F when only doing motion blur post fx in
   * panoramic camera. */
  velocity_view_tx_ = DRW_texture_pool_query_2d(UNPACK2(extent), GPU_RGBA16F, owner);

  velocity_only_fb_.ensure(GPU_ATTACHMENT_NONE,
                           GPU_ATTACHMENT_TEXTURE(velocity_camera_tx_),
                           GPU_ATTACHMENT_TEXTURE(velocity_view_tx_));
}

void Velocity::render(GPUTexture *depth_tx)
{
  DRW_stats_group_start("VelocityModule");

  GPU_framebuffer_bind(velocity_only_fb_);
  inst_.shading_passes.velocity.resolve_camera_motion(depth_tx);

  velocity_fb_.ensure(GPU_ATTACHMENT_TEXTURE(depth_tx),
                      GPU_ATTACHMENT_TEXTURE(velocity_camera_tx_),
                      GPU_ATTACHMENT_TEXTURE(velocity_view_tx_));

  GPU_framebuffer_bind(velocity_fb_);
  inst_.shading_passes.velocity.render_objects();

  DRW_stats_group_end();
}

/** \} */

}  // namespace blender::eevee
