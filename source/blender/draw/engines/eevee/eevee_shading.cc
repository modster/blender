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
 * Shading passes contain drawcalls specific to shading pipelines.
 * They are to be shared across views.
 * This file is only for shading passes. Other passes are declared in their own module.
 */

#include "eevee_instance.hh"

#include "eevee_shading.hh"

namespace blender::eevee {

/* -------------------------------------------------------------------- */
/** \name Passes
 * \{ */

void ForwardPass::sync()
{
  DRWState state = DRW_STATE_WRITE_COLOR | DRW_STATE_WRITE_DEPTH | DRW_STATE_DEPTH_LESS;
  opaque_ps_ = DRW_pass_create("Forward", state);

  DRWState state_add = DRW_STATE_WRITE_COLOR | DRW_STATE_BLEND_ADD_FULL | DRW_STATE_DEPTH_EQUAL;
  light_additional_ps_ = DRW_pass_create_instance("ForwardAddLight", opaque_ps_, state_add);
}

void ForwardPass::surface_add(Object *ob, Material *mat, int matslot)
{
  (void)mat;
  (void)matslot;
  GPUBatch *geom = DRW_cache_object_surface_get(ob);
  if (geom == nullptr) {
    return;
  }

  LightModule &lights = inst_.lights;

  GPUShader *sh = inst_.shaders.static_shader_get(MESH);
  DRWShadingGroup *grp = DRW_shgroup_create(sh, opaque_ps_);
  DRW_shgroup_uniform_block_ref(grp, "lights_block", lights.lights_ubo_ref_get());
  DRW_shgroup_uniform_block_ref(grp, "shadows_punctual_block", lights.shadows_ubo_ref_get());
  DRW_shgroup_uniform_block_ref(grp, "lights_culling_block", lights.culling_ubo_ref_get());
  DRW_shgroup_uniform_texture_ref(grp, "lights_culling_tx", lights.culling_tx_ref_get());
  DRW_shgroup_uniform_texture(grp, "utility_tx", inst_.shading_passes.utility_tx);
  DRW_shgroup_uniform_texture_ref(grp, "shadow_atlas_tx", inst_.shadows.atlas_ref_get());
  DRW_shgroup_call(grp, geom, ob);
}

void ForwardPass::render(void)
{
  for (auto index : inst_.lights.index_range()) {
    inst_.lights.bind_batch(index);

    DRW_draw_pass((index == 0) ? opaque_ps_ : light_additional_ps_);
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name DeferredLayer
 * \{ */

void DeferredLayer::sync(void)
{
  DRWState state = DRW_STATE_WRITE_COLOR | DRW_STATE_WRITE_DEPTH | DRW_STATE_DEPTH_LESS;
  gbuffer_ps_ = DRW_pass_create("GbufferLayer", state);
}

void DeferredLayer::surface_add(Object *ob)
{
  GPUBatch *geom = DRW_cache_object_surface_get(ob);
  if (geom == nullptr) {
    return;
  }

  GPUShader *sh = inst_.shaders.static_shader_get(DEFERRED_MESH);
  DRWShadingGroup *grp = DRW_shgroup_create(sh, gbuffer_ps_);
  DRW_shgroup_call(grp, geom, ob);
}

void DeferredLayer::render(GBuffer &gbuffer, GPUFrameBuffer *view_fb)
{
  if (DRW_pass_is_empty(gbuffer_ps_)) {
    return;
  }

  gbuffer.bind(CLOSURE_DIFFUSE);
  DRW_draw_pass(gbuffer_ps_);

  inst_.shading_passes.deferred.input_diffuse_data_tx_ = gbuffer.diffuse_tx;
  inst_.shading_passes.deferred.input_reflection_data_tx_ = gbuffer.reflection_tx;

  for (auto index : inst_.lights.index_range()) {
    inst_.lights.bind_batch(index);

    GPU_framebuffer_bind(view_fb);
    DRW_draw_pass(inst_.shading_passes.deferred.eval_diffuse_ps_);
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name DeferredLayer
 * \{ */

void DeferredPass::sync(void)
{
  opaque_ps_.sync();
  refraction_ps_.sync();
  volumetric_ps_.sync();

  LightModule &lights = inst_.lights;

  {
    DRWState state = DRW_STATE_WRITE_COLOR | DRW_STATE_DEPTH_GREATER | DRW_STATE_BLEND_ADD_FULL;
    eval_diffuse_ps_ = DRW_pass_create("DeferredEvalDiffuse", state);
    GPUShader *sh = inst_.shaders.static_shader_get(DEFERRED_EVAL_DIRECT);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, eval_diffuse_ps_);
    DRW_shgroup_uniform_block_ref(grp, "lights_block", lights.lights_ubo_ref_get());
    DRW_shgroup_uniform_block_ref(grp, "shadows_punctual_block", lights.shadows_ubo_ref_get());
    DRW_shgroup_uniform_block_ref(grp, "lights_culling_block", lights.culling_ubo_ref_get());
    DRW_shgroup_uniform_texture_ref(grp, "lights_culling_tx", lights.culling_tx_ref_get());
    DRW_shgroup_uniform_texture(grp, "utility_tx", inst_.shading_passes.utility_tx);
    DRW_shgroup_uniform_texture_ref(grp, "shadow_atlas_tx", inst_.shadows.atlas_ref_get());
    DRW_shgroup_uniform_texture_ref(grp, "diffuse_data_tx", &input_diffuse_data_tx_);
    DRW_shgroup_uniform_texture_ref(grp, "reflection_data_tx", &input_reflection_data_tx_);
    DRW_shgroup_uniform_texture_ref(grp, "depth_tx", &input_depth_tx_);
    DRW_shgroup_call_procedural_triangles(grp, nullptr, 3);
  }
}

void DeferredPass::surface_add(Object *ob)
{
  opaque_ps_.surface_add(ob);
}

void DeferredPass::render(GBuffer &gbuffer, GPUFrameBuffer *view_fb)
{
  input_depth_tx_ = gbuffer.depth_tx;

  opaque_ps_.render(gbuffer, view_fb);
  refraction_ps_.render(gbuffer, view_fb);
  volumetric_ps_.render(gbuffer, view_fb);

  gbuffer.render_end();
}

/** \} */

}  // namespace blender::eevee