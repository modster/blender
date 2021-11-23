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

void BackgroundPass::sync(GPUMaterial *gpumat, GPUTexture *lookdev_tx)
{
  DRWState state = DRW_STATE_WRITE_COLOR;
  background_ps_ = DRW_pass_create("Background", state);

  /* Push a matrix at the same location as the camera. */
  mat4 camera_mat;
  unit_m4(camera_mat);
  copy_v3_v3(camera_mat[3], inst_.camera.data_get().viewinv[3]);

  DRWShadingGroup *grp = DRW_shgroup_material_create(gpumat, background_ps_);
  if (lookdev_tx != nullptr) {
    /* HACK(fclem) This particular texture has been left without resource to be set here. */
    DRW_shgroup_uniform_texture(grp, "samp0", lookdev_tx);
  }
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
 * NPR materials (using Closure to RGBA) or material using ALPHA_BLEND.
 * \{ */

void ForwardPass::sync(void)
{
  {
    DRWState state = DRW_STATE_WRITE_DEPTH | DRW_STATE_DEPTH_LESS;
    prepass_ps_ = DRW_pass_create("Forward.Opaque.Prepass", state);

    state |= DRW_STATE_CULL_BACK;
    prepass_culled_ps_ = DRW_pass_create("Forward.Opaque.Prepass.Culled", state);

    DRW_pass_link(prepass_ps_, prepass_culled_ps_);
  }
  {
    DRWState state = DRW_STATE_WRITE_COLOR | DRW_STATE_DEPTH_EQUAL;
    opaque_ps_ = DRW_pass_create("Forward.Opaque", state);

    state |= DRW_STATE_CULL_BACK;
    opaque_culled_ps_ = DRW_pass_create("Forward.Opaque.Culled", state);

    DRW_pass_link(opaque_ps_, opaque_culled_ps_);
  }
  {
    DRWState state = DRW_STATE_DEPTH_LESS_EQUAL;
    transparent_ps_ = DRW_pass_create("Forward.Transparent", state);
  }
}

DRWShadingGroup *ForwardPass::material_opaque_add(::Material *blender_mat, GPUMaterial *gpumat)
{
  DRWPass *pass = (blender_mat->blend_flag & MA_BL_CULL_BACKFACE) ? opaque_culled_ps_ : opaque_ps_;
  LightModule &lights = inst_.lights;
  LightProbeModule &lightprobes = inst_.lightprobes;
  eGPUSamplerState no_interp = GPU_SAMPLER_DEFAULT;
  DRWShadingGroup *grp = DRW_shgroup_material_create(gpumat, pass);
  lights.shgroup_resources(grp);
  DRW_shgroup_uniform_block(grp, "sampling_block", inst_.sampling.ubo_get());
  DRW_shgroup_uniform_block(grp, "grids_block", lightprobes.grid_ubo_get());
  DRW_shgroup_uniform_block(grp, "cubes_block", lightprobes.cube_ubo_get());
  DRW_shgroup_uniform_block(grp, "lightprobes_info_block", lightprobes.info_ubo_get());
  DRW_shgroup_uniform_texture_ref(grp, "lightprobe_grid_tx", lightprobes.grid_tx_ref_get());
  DRW_shgroup_uniform_texture_ref(grp, "lightprobe_cube_tx", lightprobes.cube_tx_ref_get());
  DRW_shgroup_uniform_texture(grp, "utility_tx", inst_.shading_passes.utility_tx);
  /* TODO(fclem): Make this only needed if material uses it ... somehow. */
  if (true) {
    DRW_shgroup_uniform_texture_ref(
        grp, "sss_transmittance_tx", inst_.subsurface.transmittance_ref_get());
  }
  if (true) {
    DRW_shgroup_uniform_block(grp, "rt_diffuse_block", inst_.raytracing.diffuse_ubo_get());
    DRW_shgroup_uniform_block(grp, "rt_reflection_block", inst_.raytracing.reflection_ubo_get());
    DRW_shgroup_uniform_block(grp, "rt_refraction_block", inst_.raytracing.refraction_ubo_get());
    DRW_shgroup_uniform_texture_ref_ex(grp, "radiance_tx", &input_radiance_tx_, no_interp);
  }
  if (true) {
    DRW_shgroup_uniform_block(grp, "hiz_block", inst_.hiz.ubo_get());
    DRW_shgroup_uniform_texture_ref(grp, "hiz_tx", &input_hiz_tx_);
  }
  return grp;
}

DRWShadingGroup *ForwardPass::prepass_opaque_add(::Material *blender_mat, GPUMaterial *gpumat)
{
  DRWPass *pass = (blender_mat->blend_flag & MA_BL_CULL_BACKFACE) ? prepass_culled_ps_ :
                                                                    prepass_ps_;
  DRWShadingGroup *grp = DRW_shgroup_material_create(gpumat, pass);
  return grp;
}

DRWShadingGroup *ForwardPass::material_transparent_add(::Material *blender_mat,
                                                       GPUMaterial *gpumat)
{
  LightModule &lights = inst_.lights;
  LightProbeModule &lightprobes = inst_.lightprobes;
  eGPUSamplerState no_interp = GPU_SAMPLER_DEFAULT;
  DRWShadingGroup *grp = DRW_shgroup_material_create(gpumat, transparent_ps_);
  lights.shgroup_resources(grp);
  DRW_shgroup_uniform_block(grp, "sampling_block", inst_.sampling.ubo_get());
  DRW_shgroup_uniform_block(grp, "grids_block", lightprobes.grid_ubo_get());
  DRW_shgroup_uniform_block(grp, "cubes_block", lightprobes.cube_ubo_get());
  DRW_shgroup_uniform_block(grp, "lightprobes_info_block", lightprobes.info_ubo_get());
  DRW_shgroup_uniform_texture_ref(grp, "lightprobe_grid_tx", lightprobes.grid_tx_ref_get());
  DRW_shgroup_uniform_texture_ref(grp, "lightprobe_cube_tx", lightprobes.cube_tx_ref_get());
  DRW_shgroup_uniform_texture(grp, "utility_tx", inst_.shading_passes.utility_tx);
  /* TODO(fclem): Make this only needed if material uses it ... somehow. */
  if (true) {
    DRW_shgroup_uniform_texture_ref(
        grp, "sss_transmittance_tx", inst_.subsurface.transmittance_ref_get());
  }
  if (true) {
    DRW_shgroup_uniform_block(grp, "rt_diffuse_block", inst_.raytracing.diffuse_ubo_get());
    DRW_shgroup_uniform_block(grp, "rt_reflection_block", inst_.raytracing.reflection_ubo_get());
    DRW_shgroup_uniform_block(grp, "rt_refraction_block", inst_.raytracing.refraction_ubo_get());
    DRW_shgroup_uniform_texture_ref_ex(grp, "radiance_tx", &input_radiance_tx_, no_interp);
  }
  if (true) {
    DRW_shgroup_uniform_block(grp, "hiz_block", inst_.hiz.ubo_get());
    DRW_shgroup_uniform_texture_ref(grp, "hiz_tx", &input_hiz_tx_);
  }

  DRWState state_disable = DRW_STATE_WRITE_DEPTH;
  DRWState state_enable = DRW_STATE_WRITE_COLOR | DRW_STATE_BLEND_CUSTOM;
  if (blender_mat->blend_flag & MA_BL_CULL_BACKFACE) {
    state_enable |= DRW_STATE_CULL_BACK;
  }
  DRW_shgroup_state_disable(grp, state_disable);
  DRW_shgroup_state_enable(grp, state_enable);
  return grp;
}

DRWShadingGroup *ForwardPass::prepass_transparent_add(::Material *blender_mat, GPUMaterial *gpumat)
{
  if ((blender_mat->blend_flag & MA_BL_HIDE_BACKFACE) == 0) {
    return nullptr;
  }

  DRWShadingGroup *grp = DRW_shgroup_material_create(gpumat, transparent_ps_);

  DRWState state_disable = DRW_STATE_WRITE_COLOR | DRW_STATE_BLEND_CUSTOM;
  DRWState state_enable = DRW_STATE_WRITE_DEPTH;
  if (blender_mat->blend_flag & MA_BL_CULL_BACKFACE) {
    state_enable |= DRW_STATE_CULL_BACK;
  }
  DRW_shgroup_state_disable(grp, state_disable);
  DRW_shgroup_state_enable(grp, state_enable);
  return grp;
}

void ForwardPass::render(GBuffer &gbuffer, HiZBuffer &hiz, GPUFrameBuffer *view_fb)
{
  if (inst_.raytracing.enabled()) {
    ivec2 extent = {GPU_texture_width(gbuffer.depth_tx), GPU_texture_height(gbuffer.depth_tx)};
    /* Reuse texture. */
    gbuffer.ray_radiance_tx.acquire_tmp(UNPACK2(extent), GPU_RGBA16F, gbuffer.owner);
    /* Copy combined buffer so we can sample from it. */
    GPU_texture_copy(gbuffer.ray_radiance_tx, gbuffer.combined_tx);

    input_radiance_tx_ = gbuffer.ray_radiance_tx;

    hiz.prepare(gbuffer.depth_tx);
    /* TODO(fclem): Avoid this if possible. */
    hiz.update(gbuffer.depth_tx);

    input_hiz_tx_ = hiz.texture_get();

    GPU_framebuffer_bind(view_fb);
  }

  DRW_draw_pass(prepass_ps_);
  DRW_draw_pass(opaque_ps_);

  /* TODO(fclem) This is suboptimal. We could sort during sync. */
  /* FIXME(fclem) This wont work for panoramic, where we need
   * to sort by distance to camera, not by z. */
  DRW_pass_sort_shgroup_z(transparent_ps_);
  DRW_draw_pass(transparent_ps_);

  if (inst_.raytracing.enabled()) {
    gbuffer.ray_radiance_tx.release_tmp();
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name DeferredLayer
 * \{ */

void DeferredLayer::sync(void)
{
  {
    DRWState state = DRW_STATE_WRITE_DEPTH | DRW_STATE_DEPTH_LESS;
    prepass_ps_ = DRW_pass_create("Gbuffer.Prepass", state);

    state |= DRW_STATE_CULL_BACK;
    prepass_culled_ps_ = DRW_pass_create("Gbuffer.Prepass.Culled", state);

    DRW_pass_link(prepass_ps_, prepass_culled_ps_);
  }
  {
    DRWState state = DRW_STATE_WRITE_COLOR | DRW_STATE_DEPTH_EQUAL | DRW_STATE_STENCIL_ALWAYS |
                     DRW_STATE_WRITE_STENCIL;
    gbuffer_ps_ = DRW_pass_create("Gbuffer", state);

    state |= DRW_STATE_CULL_BACK;
    gbuffer_culled_ps_ = DRW_pass_create("Gbuffer.Culled", state);

    DRW_pass_link(gbuffer_ps_, gbuffer_culled_ps_);
  }
  {
    DRWState state = DRW_STATE_WRITE_COLOR | DRW_STATE_WRITE_DEPTH | DRW_STATE_DEPTH_LESS_EQUAL |
                     DRW_STATE_CULL_BACK | DRW_STATE_STENCIL_ALWAYS | DRW_STATE_WRITE_STENCIL;
    volume_ps_ = DRW_pass_create("VolumesHeterogeneous", state);
  }
}

DRWShadingGroup *DeferredLayer::material_add(::Material *blender_mat, GPUMaterial *gpumat)
{
  /* TODO/OPTI(fclem) Set the right mask for each effect based on gpumat flags. */
  uint stencil_mask = CLOSURE_DIFFUSE | CLOSURE_SSS | CLOSURE_REFLECTION | CLOSURE_TRANSPARENCY |
                      CLOSURE_EMISSION | CLOSURE_REFRACTION;
  DRWPass *pass = (blender_mat->blend_flag & MA_BL_CULL_BACKFACE) ? gbuffer_culled_ps_ :
                                                                    gbuffer_ps_;
  DRWShadingGroup *grp = DRW_shgroup_material_create(gpumat, pass);
  DRW_shgroup_uniform_block(grp, "sampling_block", inst_.sampling.ubo_get());
  DRW_shgroup_uniform_texture(grp, "utility_tx", inst_.shading_passes.utility_tx);
  DRW_shgroup_stencil_set(grp, stencil_mask, 0xFF, 0xFF);
  return grp;
}

DRWShadingGroup *DeferredLayer::prepass_add(::Material *blender_mat, GPUMaterial *gpumat)
{
  DRWPass *pass = (blender_mat->blend_flag & MA_BL_CULL_BACKFACE) ? prepass_culled_ps_ :
                                                                    prepass_ps_;
  DRWShadingGroup *grp = DRW_shgroup_material_create(gpumat, pass);
  DRW_shgroup_uniform_block(grp, "sampling_block", inst_.sampling.ubo_get());
  return grp;
}

void DeferredLayer::volume_add(Object *ob)
{
  LightModule &lights = inst_.lights;
  DeferredPass &deferred_pass = inst_.shading_passes.deferred;

  GPUShader *sh = inst_.shaders.static_shader_get(DEFERRED_VOLUME);
  DRWShadingGroup *grp = DRW_shgroup_create(sh, volume_ps_);
  lights.shgroup_resources(grp);
  DRW_shgroup_uniform_texture_ref(grp, "depth_max_tx", &deferred_pass.input_depth_behind_tx_);
  DRW_shgroup_uniform_texture(grp, "utility_tx", inst_.shading_passes.utility_tx);
  DRW_shgroup_stencil_set(grp, CLOSURE_VOLUME | CLOSURE_TRANSPARENCY, 0xFF, 0xFF);
  DRW_shgroup_call(grp, DRW_cache_cube_get(), ob);
}

void DeferredLayer::render(GBuffer &gbuffer,
                           HiZBuffer &hiz_front,
                           HiZBuffer &hiz_back,
                           RaytraceBuffer &rt_buffer,
                           GPUFrameBuffer *view_fb)
{
  DeferredPass &deferred_pass = inst_.shading_passes.deferred;

  const bool no_surfaces = DRW_pass_is_empty(gbuffer_ps_);
  const bool no_volumes = DRW_pass_is_empty(volume_ps_);
  if (no_surfaces && no_volumes) {
    return;
  }
  /* TODO(fclem): detect these cases. */
  const bool use_diffuse = true;
  const bool use_subsurface = true;
  const bool use_transparency = true;
  const bool use_holdout = true;
  const bool use_refraction = true;
  const bool use_glossy = true;
  const bool use_ao = false;

  gbuffer.prepare((eClosureBits)0xFFFFFFFFu);
  if (use_ao || use_glossy || use_diffuse) {
    hiz_front.prepare(gbuffer.depth_tx);
  }
  if (use_refraction) {
    hiz_back.prepare(gbuffer.depth_tx);
  }

  update_pass_inputs(gbuffer, hiz_front, hiz_back);

  if (use_refraction) {
    /* TODO(fclem) Only update if needed.
     * i.e: No need when SSR from previous layer has already updated hiz. */
    hiz_back.update(gbuffer.depth_tx);
  }

  gbuffer.bind();

  if (!no_surfaces) {
    DRW_draw_pass(prepass_ps_);

    /* TODO(fclem): Ambient Occlusion texture node. */
    if (use_ao) {
      hiz_front.update(gbuffer.depth_tx);
      gbuffer.bind();
    }

    DRW_draw_pass(gbuffer_ps_);
  }

  if (!no_volumes) {
    // gbuffer.copy_depth_behind();
    // deferred_pass.input_depth_behind_tx_ = gbuffer.depth_behind_tx;

    gbuffer.bind_volume();
    DRW_draw_pass(volume_ps_);
  }

  if (use_holdout) {
    gbuffer.bind_holdout();
    DRW_draw_pass(deferred_pass.eval_holdout_ps_);
  }

  /* TODO(fclem) We could bypass update if ao already updated it and if there is no volume. */
  hiz_front.update(gbuffer.depth_tx);

  if (!no_surfaces && use_refraction) {
    /* Launch and shade refraction rays before transparency changes the combined pass. */
    rt_buffer.trace(CLOSURE_REFRACTION, gbuffer, hiz_back, hiz_front);
  }

  GPU_framebuffer_bind(view_fb);
  if (use_transparency) {
    DRW_draw_pass(deferred_pass.eval_transparency_ps_);
  }

  gbuffer.clear_radiance();

  if (!no_surfaces && use_refraction) {
    rt_buffer.denoise(CLOSURE_REFRACTION);
    rt_buffer.resolve(CLOSURE_REFRACTION, gbuffer);
  }

  if (!no_volumes) {
    /* TODO(fclem) volume fb. */
    GPU_framebuffer_bind(view_fb);
    DRW_draw_pass(deferred_pass.eval_volume_homogeneous_ps_);
  }

  if (!no_surfaces) {
    gbuffer.bind_radiance();
    DRW_draw_pass(deferred_pass.eval_direct_ps_);

    if (use_diffuse) {
      rt_buffer.trace(CLOSURE_DIFFUSE, gbuffer, hiz_front, hiz_front);
      rt_buffer.denoise(CLOSURE_DIFFUSE);
      rt_buffer.resolve(CLOSURE_DIFFUSE, gbuffer);
    }

    if (use_subsurface) {
      GPU_framebuffer_bind(view_fb);
      DRW_draw_pass(deferred_pass.eval_subsurface_ps_);
    }

    if (use_glossy) {
      rt_buffer.trace(CLOSURE_REFLECTION, gbuffer, hiz_front, hiz_front);
      rt_buffer.denoise(CLOSURE_REFLECTION);
      rt_buffer.resolve(CLOSURE_REFLECTION, gbuffer);
    }
  }
}

void DeferredLayer::update_pass_inputs(GBuffer &gbuffer, HiZBuffer &hiz_front, HiZBuffer &hiz_back)
{
  DeferredPass &deferred_pass = inst_.shading_passes.deferred;
  deferred_pass.input_combined_tx_ = gbuffer.combined_tx;
  deferred_pass.input_emission_data_tx_ = gbuffer.emission_tx;
  deferred_pass.input_transmit_color_tx_ = gbuffer.transmit_color_tx;
  deferred_pass.input_transmit_normal_tx_ = gbuffer.transmit_normal_tx;
  deferred_pass.input_transmit_data_tx_ = gbuffer.transmit_data_tx;
  deferred_pass.input_reflect_color_tx_ = gbuffer.reflect_color_tx;
  deferred_pass.input_reflect_normal_tx_ = gbuffer.reflect_normal_tx;
  deferred_pass.input_diffuse_tx_ = gbuffer.diffuse_tx;
  deferred_pass.input_transparency_data_tx_ = gbuffer.transparency_tx;
  deferred_pass.input_volume_data_tx_ = gbuffer.volume_tx;
  deferred_pass.input_hiz_front_tx_ = hiz_front.texture_get();
  deferred_pass.input_hiz_back_tx_ = hiz_back.texture_get();
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
  LightProbeModule &lightprobes = inst_.lightprobes;

  eGPUSamplerState no_interp = GPU_SAMPLER_DEFAULT;

  {
    DRWState state = DRW_STATE_WRITE_COLOR | DRW_STATE_STENCIL_NEQUAL | DRW_STATE_BLEND_ADD_FULL;
    eval_direct_ps_ = DRW_pass_create("DeferredDirect", state);
    GPUShader *sh = inst_.shaders.static_shader_get(DEFERRED_EVAL_DIRECT);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, eval_direct_ps_);
    lights.shgroup_resources(grp);
    DRW_shgroup_uniform_block(grp, "sampling_block", inst_.sampling.ubo_get());
    DRW_shgroup_uniform_block(grp, "grids_block", lightprobes.grid_ubo_get());
    DRW_shgroup_uniform_block(grp, "cubes_block", lightprobes.cube_ubo_get());
    DRW_shgroup_uniform_block(grp, "lightprobes_info_block", lightprobes.info_ubo_get());
    DRW_shgroup_uniform_texture_ref(grp, "lightprobe_grid_tx", lightprobes.grid_tx_ref_get());
    DRW_shgroup_uniform_texture_ref(grp, "lightprobe_cube_tx", lightprobes.cube_tx_ref_get());
    DRW_shgroup_uniform_texture(grp, "utility_tx", inst_.shading_passes.utility_tx);
    DRW_shgroup_uniform_texture_ref_ex(
        grp, "emission_data_tx", &input_emission_data_tx_, no_interp);
    DRW_shgroup_uniform_texture_ref_ex(
        grp, "transmit_color_tx", &input_transmit_color_tx_, no_interp);
    DRW_shgroup_uniform_texture_ref_ex(
        grp, "transmit_normal_tx", &input_transmit_normal_tx_, no_interp);
    DRW_shgroup_uniform_texture_ref_ex(
        grp, "transmit_data_tx", &input_transmit_data_tx_, no_interp);
    DRW_shgroup_uniform_texture_ref_ex(
        grp, "reflect_color_tx", &input_reflect_color_tx_, no_interp);
    DRW_shgroup_uniform_texture_ref_ex(
        grp, "reflect_normal_tx", &input_reflect_normal_tx_, no_interp);
    DRW_shgroup_uniform_texture_ref(grp, "hiz_tx", &input_hiz_front_tx_);
    DRW_shgroup_uniform_texture_ref(
        grp, "sss_transmittance_tx", inst_.subsurface.transmittance_ref_get());
    DRW_shgroup_stencil_set(
        grp, 0x0, 0x0, CLOSURE_DIFFUSE | CLOSURE_REFLECTION | CLOSURE_EMISSION);
    DRW_shgroup_call_procedural_triangles(grp, nullptr, 1);
  }
  {
    DRWState state = DRW_STATE_WRITE_COLOR | DRW_STATE_STENCIL_EQUAL | DRW_STATE_BLEND_ADD_FULL;
    eval_subsurface_ps_ = DRW_pass_create("DeferredSubsurface", state);
    GPUShader *sh = inst_.shaders.static_shader_get(SUBSURFACE_EVAL);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, eval_subsurface_ps_);
    DRW_shgroup_uniform_block(grp, "subsurface_block", inst_.subsurface.ubo_get());
    DRW_shgroup_uniform_block(grp, "sampling_block", inst_.sampling.ubo_get());
    DRW_shgroup_uniform_block(grp, "hiz_block", inst_.hiz.ubo_get());
    DRW_shgroup_uniform_texture_ref(grp, "hiz_tx", &input_hiz_front_tx_);
    DRW_shgroup_uniform_texture_ref_ex(grp, "radiance_tx", &input_diffuse_tx_, no_interp);
    DRW_shgroup_uniform_texture_ref_ex(
        grp, "transmit_color_tx", &input_transmit_color_tx_, no_interp);
    DRW_shgroup_uniform_texture_ref_ex(
        grp, "transmit_normal_tx", &input_transmit_normal_tx_, no_interp);
    DRW_shgroup_uniform_texture_ref_ex(
        grp, "transmit_data_tx", &input_transmit_data_tx_, no_interp);
    DRW_shgroup_stencil_set(grp, 0x0, 0xFF, CLOSURE_SSS);
    DRW_shgroup_call_procedural_triangles(grp, nullptr, 1);
  }
  {
    DRWState state = DRW_STATE_WRITE_COLOR | DRW_STATE_STENCIL_NEQUAL | DRW_STATE_BLEND_ADD_FULL;
    eval_volume_homogeneous_ps_ = DRW_pass_create("DeferredVolume", state);
    GPUShader *sh = inst_.shaders.static_shader_get(DEFERRED_EVAL_VOLUME);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, eval_volume_homogeneous_ps_);
    lights.shgroup_resources(grp);
    DRW_shgroup_uniform_texture(grp, "utility_tx", inst_.shading_passes.utility_tx);
    DRW_shgroup_uniform_texture_ref_ex(
        grp, "transparency_data_tx", &input_transparency_data_tx_, no_interp);
    DRW_shgroup_uniform_texture_ref_ex(grp, "volume_data_tx", &input_volume_data_tx_, no_interp);
    DRW_shgroup_uniform_texture_ref(grp, "hiz_tx", &input_hiz_front_tx_);
    DRW_shgroup_stencil_set(grp, 0x0, 0x0, CLOSURE_VOLUME);
    DRW_shgroup_call_procedural_triangles(grp, nullptr, 1);
  }
  {
    DRWState state = DRW_STATE_WRITE_COLOR | DRW_STATE_STENCIL_NEQUAL | DRW_STATE_BLEND_MUL;
    eval_transparency_ps_ = DRW_pass_create("DeferredTransparency", state);
    GPUShader *sh = inst_.shaders.static_shader_get(DEFERRED_EVAL_TRANSPARENT);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, eval_transparency_ps_);
    DRW_shgroup_uniform_texture_ref(grp, "transparency_data_tx", &input_transparency_data_tx_);
    DRW_shgroup_uniform_texture_ref_ex(grp, "volume_data_tx", &input_volume_data_tx_, no_interp);
    DRW_shgroup_stencil_set(grp, 0x0, 0x0, CLOSURE_TRANSPARENCY);
    DRW_shgroup_call_procedural_triangles(grp, nullptr, 1);
  }
  {
    DRWState state = DRW_STATE_WRITE_COLOR | DRW_STATE_STENCIL_NEQUAL;
    eval_holdout_ps_ = DRW_pass_create("DeferredHoldout", state);
    GPUShader *sh = inst_.shaders.static_shader_get(DEFERRED_EVAL_HOLDOUT);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, eval_volume_homogeneous_ps_);
    DRW_shgroup_uniform_texture_ref(grp, "combined_tx", &input_combined_tx_);
    DRW_shgroup_uniform_texture_ref(grp, "transparency_data_tx", &input_transparency_data_tx_);
    DRW_shgroup_stencil_set(grp, 0x0, 0x0, CLOSURE_TRANSPARENCY);
    DRW_shgroup_call_procedural_triangles(grp, nullptr, 1);
  }
}

DRWShadingGroup *DeferredPass::material_add(::Material *material, GPUMaterial *gpumat)
{
  if (material->blend_flag & MA_BL_SS_REFRACTION) {
    return refraction_layer_.material_add(material, gpumat);
  }
  else {
    return opaque_layer_.material_add(material, gpumat);
  }
}

DRWShadingGroup *DeferredPass::prepass_add(::Material *material, GPUMaterial *gpumat)
{
  if (material->blend_flag & MA_BL_SS_REFRACTION) {
    return refraction_layer_.prepass_add(material, gpumat);
  }
  else {
    return opaque_layer_.prepass_add(material, gpumat);
  }
}

void DeferredPass::volume_add(Object *ob)
{
  volumetric_layer_.volume_add(ob);
}

void DeferredPass::render(const DRWView *drw_view,
                          GBuffer &gbuffer,
                          HiZBuffer &hiz_front,
                          HiZBuffer &hiz_back,
                          RaytraceBuffer &rt_buffer_opaque_,
                          RaytraceBuffer &rt_buffer_refract_,
                          GPUFrameBuffer *view_fb)
{
  DRW_stats_group_start("OpaqueLayer");
  opaque_layer_.render(gbuffer, hiz_front, hiz_back, rt_buffer_opaque_, view_fb);
  DRW_stats_group_end();

  DRW_stats_group_start("RefractionLayer");
  refraction_layer_.render(gbuffer, hiz_front, hiz_back, rt_buffer_refract_, view_fb);
  DRW_stats_group_end();

  /* NOTE(fclem): Reuse the same rtbuffer as refraction but should not use it. */
  DRW_stats_group_start("VolumetricLayer");
  volumetric_layer_.render(gbuffer, hiz_front, hiz_back, rt_buffer_refract_, view_fb);
  DRW_stats_group_end();

  gbuffer.render_end();
  rt_buffer_opaque_.render_end(drw_view);
  rt_buffer_refract_.render_end(drw_view);
}

/** \} */

}  // namespace blender::eevee