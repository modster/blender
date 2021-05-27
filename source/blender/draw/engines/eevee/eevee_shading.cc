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
/** \name Background Pass
 *
 * \{ */

void BackgroundPass::sync(GPUMaterial *gpumat)
{
  DRWState state = DRW_STATE_WRITE_COLOR;
  background_ps_ = DRW_pass_create("Background", state);

  /* Push a matrix at the same location as the camera. */
  mat4 camera_mat;
  unit_m4(camera_mat);
  copy_v3_v3(camera_mat[3], inst_.camera.data_get().viewinv[3]);

  DRWShadingGroup *grp = DRW_shgroup_material_create(gpumat, background_ps_);
  DRW_shgroup_call_obmat(grp, DRW_cache_fullscreen_quad_get(), camera_mat);
}

void BackgroundPass::render(void)
{
  DRW_draw_pass(background_ps_);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Forward Pass
 *
 * Handles alpha blended surfaces and NPR materials (using Closure to RGBA).
 * \{ */

void ForwardPass::sync()
{
  DRWState state = DRW_STATE_WRITE_COLOR | DRW_STATE_WRITE_DEPTH | DRW_STATE_DEPTH_LESS;
  opaque_ps_ = DRW_pass_create("Forward", state);

  DRWState state_add = DRW_STATE_WRITE_COLOR | DRW_STATE_BLEND_ADD_FULL | DRW_STATE_DEPTH_EQUAL;
  light_additional_ps_ = DRW_pass_create_instance("ForwardAddLight", opaque_ps_, state_add);
}

DRWShadingGroup *ForwardPass::material_add(GPUMaterial *gpumat)
{
  LightModule &lights = inst_.lights;
  DRWShadingGroup *grp = DRW_shgroup_material_create(gpumat, opaque_ps_);
  DRW_shgroup_uniform_block_ref(grp, "lights_block", lights.lights_ubo_ref_get());
  DRW_shgroup_uniform_block_ref(grp, "shadows_punctual_block", lights.shadows_ubo_ref_get());
  DRW_shgroup_uniform_block_ref(grp, "lights_culling_block", lights.culling_ubo_ref_get());
  DRW_shgroup_uniform_texture_ref(grp, "lights_culling_tx", lights.culling_tx_ref_get());
  DRW_shgroup_uniform_texture(grp, "utility_tx", inst_.shading_passes.utility_tx);
  DRW_shgroup_uniform_texture_ref(grp, "shadow_atlas_tx", inst_.shadows.atlas_ref_get());
  return grp;
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
  {
    DRWState state = DRW_STATE_WRITE_COLOR | DRW_STATE_WRITE_DEPTH | DRW_STATE_DEPTH_LESS |
                     DRW_STATE_STENCIL_ALWAYS | DRW_STATE_WRITE_STENCIL;
    gbuffer_ps_ = DRW_pass_create("Gbuffer", state);
  }
  {
    DRWState state = DRW_STATE_WRITE_COLOR | DRW_STATE_WRITE_DEPTH | DRW_STATE_DEPTH_LESS_EQUAL |
                     DRW_STATE_CULL_BACK | DRW_STATE_STENCIL_ALWAYS | DRW_STATE_WRITE_STENCIL;
    volume_ps_ = DRW_pass_create("VolumesHeterogeneous", state);
  }
}

DRWShadingGroup *DeferredLayer::material_add(GPUMaterial *gpumat)
{
  uint stencil_mask = CLOSURE_DIFFUSE | CLOSURE_REFLECTION | CLOSURE_TRANSPARENCY |
                      CLOSURE_EMISSION;
  DRWShadingGroup *grp = DRW_shgroup_material_create(gpumat, gbuffer_ps_);
  DRW_shgroup_uniform_block(grp, "sampling_block", inst_.sampling.ubo_get());
  DRW_shgroup_uniform_texture(grp, "utility_tx", inst_.shading_passes.utility_tx);
  DRW_shgroup_stencil_set(grp, stencil_mask, 0xFF, 0xFF);
  return grp;
}

void DeferredLayer::volume_add(Object *ob)
{
  LightModule &lights = inst_.lights;
  DeferredPass &deferred_pass = inst_.shading_passes.deferred;

  GPUShader *sh = inst_.shaders.static_shader_get(DEFERRED_VOLUME);
  DRWShadingGroup *grp = DRW_shgroup_create(sh, volume_ps_);
  DRW_shgroup_uniform_block_ref(grp, "lights_block", lights.lights_ubo_ref_get());
  DRW_shgroup_uniform_block_ref(grp, "shadows_punctual_block", lights.shadows_ubo_ref_get());
  DRW_shgroup_uniform_block_ref(grp, "lights_culling_block", lights.culling_ubo_ref_get());
  DRW_shgroup_uniform_texture_ref(grp, "lights_culling_tx", lights.culling_tx_ref_get());
  DRW_shgroup_uniform_texture_ref(grp, "depth_max_tx", &deferred_pass.input_depth_behind_tx_);
  DRW_shgroup_uniform_texture(grp, "utility_tx", inst_.shading_passes.utility_tx);
  DRW_shgroup_stencil_set(grp, CLOSURE_VOLUME | CLOSURE_TRANSPARENCY, 0xFF, 0xFF);
  DRW_shgroup_call(grp, DRW_cache_cube_get(), ob);
}

void DeferredLayer::render(GBuffer &gbuffer, GPUFrameBuffer *view_fb)
{
  const bool no_surfaces = DRW_pass_is_empty(gbuffer_ps_);
  const bool no_volumes = DRW_pass_is_empty(volume_ps_);
  if (no_surfaces && no_volumes) {
    return;
  }

  gbuffer.bind(CLOSURE_DIFFUSE);

  if (!no_surfaces) {
    DRW_draw_pass(gbuffer_ps_);
  }

  DeferredPass &deferred_pass = inst_.shading_passes.deferred;

  deferred_pass.input_emission_data_tx_ = gbuffer.emission_tx;
  deferred_pass.input_diffuse_data_tx_ = gbuffer.diffuse_tx;
  deferred_pass.input_reflection_data_tx_ = gbuffer.reflection_tx;

  if (!no_volumes) {
    gbuffer.copy_depth_behind();
    deferred_pass.input_depth_behind_tx_ = gbuffer.depth_behind_tx;
  }

  if (!no_volumes) {
    for (auto index : inst_.lights.index_range()) {
      inst_.lights.bind_batch(index);

      gbuffer.bind_volume();
      DRW_draw_pass(volume_ps_);
    }
  }

  if (true) {
    gbuffer.bind_holdout();
    DRW_draw_pass(deferred_pass.eval_holdout_ps_);
  }

  GPU_framebuffer_bind(view_fb);

  if (true) {
    DRW_draw_pass(deferred_pass.eval_transparency_ps_);
  }

  for (auto index : inst_.lights.index_range()) {
    inst_.lights.bind_batch(index);

    if (!no_volumes) {
      DRW_draw_pass(deferred_pass.eval_volume_homogeneous_ps_);
    }
    if (!no_surfaces) {
      DRW_draw_pass(deferred_pass.eval_diffuse_ps_);
    }
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name DeferredLayer
 * \{ */

void DeferredPass::sync(void)
{
  opaque_layer_.sync();
  refraction_layer_.sync();
  volumetric_layer_.sync();

  LightModule &lights = inst_.lights;

  {
    DRWState state = DRW_STATE_WRITE_COLOR | DRW_STATE_STENCIL_NEQUAL | DRW_STATE_BLEND_ADD_FULL;
    eval_diffuse_ps_ = DRW_pass_create("DeferredDirect", state);
    GPUShader *sh = inst_.shaders.static_shader_get(DEFERRED_EVAL_DIRECT);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, eval_diffuse_ps_);
    DRW_shgroup_uniform_block_ref(grp, "lights_block", lights.lights_ubo_ref_get());
    DRW_shgroup_uniform_block_ref(grp, "shadows_punctual_block", lights.shadows_ubo_ref_get());
    DRW_shgroup_uniform_block_ref(grp, "lights_culling_block", lights.culling_ubo_ref_get());
    DRW_shgroup_uniform_texture_ref(grp, "lights_culling_tx", lights.culling_tx_ref_get());
    DRW_shgroup_uniform_texture(grp, "utility_tx", inst_.shading_passes.utility_tx);
    DRW_shgroup_uniform_texture_ref(grp, "shadow_atlas_tx", inst_.shadows.atlas_ref_get());
    DRW_shgroup_uniform_texture_ref(grp, "emission_data_tx", &input_emission_data_tx_);
    DRW_shgroup_uniform_texture_ref(grp, "diffuse_data_tx", &input_diffuse_data_tx_);
    DRW_shgroup_uniform_texture_ref(grp, "reflection_data_tx", &input_reflection_data_tx_);
    DRW_shgroup_uniform_texture_ref(grp, "depth_tx", &input_depth_tx_);
    DRW_shgroup_stencil_set(
        grp, 0x0, 0x0, CLOSURE_DIFFUSE | CLOSURE_REFLECTION | CLOSURE_EMISSION);
    DRW_shgroup_call_procedural_triangles(grp, nullptr, 1);
  }
  {
    DRWState state = DRW_STATE_WRITE_COLOR | DRW_STATE_STENCIL_NEQUAL | DRW_STATE_BLEND_ADD_FULL;
    eval_volume_homogeneous_ps_ = DRW_pass_create("DeferredVolume", state);
    GPUShader *sh = inst_.shaders.static_shader_get(DEFERRED_EVAL_VOLUME);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, eval_volume_homogeneous_ps_);
    DRW_shgroup_uniform_block_ref(grp, "lights_block", lights.lights_ubo_ref_get());
    DRW_shgroup_uniform_block_ref(grp, "shadows_punctual_block", lights.shadows_ubo_ref_get());
    DRW_shgroup_uniform_block_ref(grp, "lights_culling_block", lights.culling_ubo_ref_get());
    DRW_shgroup_uniform_texture_ref(grp, "lights_culling_tx", lights.culling_tx_ref_get());
    DRW_shgroup_uniform_texture(grp, "utility_tx", inst_.shading_passes.utility_tx);
    DRW_shgroup_uniform_texture_ref(grp, "shadow_atlas_tx", inst_.shadows.atlas_ref_get());
    DRW_shgroup_uniform_texture_ref(grp, "transparency_data_tx", &input_transparency_data_tx_);
    DRW_shgroup_uniform_texture_ref(grp, "volume_data_tx", &input_volume_data_tx_);
    DRW_shgroup_uniform_texture_ref(grp, "depth_tx", &input_depth_tx_);
    DRW_shgroup_stencil_set(grp, 0x0, 0x0, CLOSURE_VOLUME);
    DRW_shgroup_call_procedural_triangles(grp, nullptr, 1);
  }
  {
    DRWState state = DRW_STATE_WRITE_COLOR | DRW_STATE_STENCIL_NEQUAL | DRW_STATE_BLEND_MUL;
    eval_transparency_ps_ = DRW_pass_create("DeferredTransparency", state);
    GPUShader *sh = inst_.shaders.static_shader_get(DEFERRED_EVAL_TRANSPARENT);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, eval_transparency_ps_);
    DRW_shgroup_uniform_texture_ref(grp, "transparency_data_tx", &input_transparency_data_tx_);
    DRW_shgroup_uniform_texture_ref(grp, "volume_data_tx", &input_volume_data_tx_);
    DRW_shgroup_uniform_texture_ref(grp, "depth_tx", &input_depth_tx_);
    DRW_shgroup_stencil_set(grp, 0x0, 0x0, CLOSURE_TRANSPARENCY);
    DRW_shgroup_call_procedural_triangles(grp, nullptr, 1);
  }
  {
    DRWState state = DRW_STATE_WRITE_COLOR | DRW_STATE_STENCIL_NEQUAL;
    eval_holdout_ps_ = DRW_pass_create("DeferredHoldout", state);
    GPUShader *sh = inst_.shaders.static_shader_get(DEFERRED_EVAL_HOLDOUT);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, eval_volume_homogeneous_ps_);
    DRW_shgroup_uniform_texture_ref(grp, "combined_tx", &input_combined_tx);
    DRW_shgroup_uniform_texture_ref(grp, "transparency_data_tx", &input_transparency_data_tx_);
    DRW_shgroup_stencil_set(grp, 0x0, 0x0, CLOSURE_TRANSPARENCY);
    DRW_shgroup_call_procedural_triangles(grp, nullptr, 1);
  }
}

DRWShadingGroup *DeferredPass::material_add(::Material *material, GPUMaterial *gpumat)
{
  if (material->blend_flag & MA_BL_SS_REFRACTION) {
    return refraction_layer_.material_add(gpumat);
  }
  else {
    return opaque_layer_.material_add(gpumat);
  }
}

void DeferredPass::volume_add(Object *ob)
{
  volumetric_layer_.volume_add(ob);
}

void DeferredPass::render(GBuffer &gbuffer, GPUFrameBuffer *view_fb)
{
  input_combined_tx = gbuffer.combined_tx;
  input_depth_tx_ = gbuffer.depth_tx;

  /* TODO. Remove. */
  gbuffer.bind(CLOSURE_DIFFUSE);

  input_emission_data_tx_ = gbuffer.emission_tx;
  input_diffuse_data_tx_ = gbuffer.diffuse_tx;
  input_reflection_data_tx_ = gbuffer.reflection_tx;
  input_transparency_data_tx_ = gbuffer.transparency_tx;
  input_volume_data_tx_ = gbuffer.volume_tx;

  // gbuffer.clear();

  opaque_layer_.render(gbuffer, view_fb);
  refraction_layer_.render(gbuffer, view_fb);
  volumetric_layer_.render(gbuffer, view_fb);

  gbuffer.render_end();
}

/** \} */

}  // namespace blender::eevee