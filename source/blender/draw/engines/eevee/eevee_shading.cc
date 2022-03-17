/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2021 Blender Foundation.
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
  float4x4 camera_mat = float4x4::identity();
  copy_v3_v3(camera_mat[3], inst_.camera.data_get().viewinv[3]);

  DRWShadingGroup *grp = DRW_shgroup_material_create(gpumat, background_ps_);
  DRW_shgroup_uniform_texture(grp, "utility_tx", inst_.shading_passes.utility_tx);
  if (lookdev_tx != nullptr) {
    /* HACK(fclem) This particular texture has been left without resource to be set here. */
    DRW_shgroup_uniform_texture(grp, "samp0", lookdev_tx);
  }
  DRW_shgroup_call_obmat(grp, DRW_cache_fullscreen_quad_get(), camera_mat.ptr());
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
  RaytracingModule &raytracing = inst_.raytracing;
  eGPUSamplerState no_interp = GPU_SAMPLER_DEFAULT;
  DRWShadingGroup *grp = DRW_shgroup_material_create(gpumat, pass);
  lights.shgroup_resources(grp);
  DRW_shgroup_uniform_block(grp, "sampling_buf", inst_.sampling.ubo_get());
  DRW_shgroup_uniform_block(grp, "grids_buf", lightprobes.grid_ubo_get());
  DRW_shgroup_uniform_block(grp, "cubes_buf", lightprobes.cube_ubo_get());
  DRW_shgroup_uniform_block(grp, "probes_buf", lightprobes.info_ubo_get());
  DRW_shgroup_uniform_texture_ref(grp, "lightprobe_grid_tx", lightprobes.grid_tx_ref_get());
  DRW_shgroup_uniform_texture_ref(grp, "lightprobe_cube_tx", lightprobes.cube_tx_ref_get());
  DRW_shgroup_uniform_texture(grp, "utility_tx", inst_.shading_passes.utility_tx);
  /* TODO(fclem): Make this only needed if material uses it ... somehow. */
  if (true) {
    DRW_shgroup_uniform_texture_ref(
        grp, "sss_transmittance_tx", inst_.subsurface.transmittance_ref_get());
  }
  if (raytracing.enabled()) {
    DRW_shgroup_uniform_block(grp, "rt_diffuse_buf", raytracing.diffuse_data);
    DRW_shgroup_uniform_block(grp, "rt_reflection_buf", raytracing.reflection_data);
    DRW_shgroup_uniform_block(grp, "rt_refraction_buf", raytracing.refraction_data);
    DRW_shgroup_uniform_texture_ref_ex(grp, "radiance_tx", &input_screen_radiance_tx_, no_interp);
  }
  if (raytracing.enabled()) {
    DRW_shgroup_uniform_block(grp, "hiz_buf", inst_.hiz.ubo_get());
    DRW_shgroup_uniform_texture_ref(grp, "hiz_tx", inst_.hiz_front.texture_ref_get());
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
  RaytracingModule &raytracing = inst_.raytracing;
  eGPUSamplerState no_interp = GPU_SAMPLER_DEFAULT;
  DRWShadingGroup *grp = DRW_shgroup_material_create(gpumat, transparent_ps_);
  lights.shgroup_resources(grp);
  DRW_shgroup_uniform_block(grp, "sampling_buf", inst_.sampling.ubo_get());
  DRW_shgroup_uniform_block(grp, "grids_buf", lightprobes.grid_ubo_get());
  DRW_shgroup_uniform_block(grp, "cubes_buf", lightprobes.cube_ubo_get());
  DRW_shgroup_uniform_block(grp, "probes_buf", lightprobes.info_ubo_get());
  DRW_shgroup_uniform_texture_ref(grp, "lightprobe_grid_tx", lightprobes.grid_tx_ref_get());
  DRW_shgroup_uniform_texture_ref(grp, "lightprobe_cube_tx", lightprobes.cube_tx_ref_get());
  DRW_shgroup_uniform_texture(grp, "utility_tx", inst_.shading_passes.utility_tx);
  /* TODO(fclem): Make this only needed if material uses it ... somehow. */
  if (true) {
    DRW_shgroup_uniform_texture_ref(
        grp, "sss_transmittance_tx", inst_.subsurface.transmittance_ref_get());
  }
  if (raytracing.enabled()) {
    DRW_shgroup_uniform_block(grp, "rt_diffuse_buf", raytracing.diffuse_data);
    DRW_shgroup_uniform_block(grp, "rt_reflection_buf", raytracing.reflection_data);
    DRW_shgroup_uniform_block(grp, "rt_refraction_buf", raytracing.refraction_data);
    DRW_shgroup_uniform_texture_ref_ex(
        grp, "rt_radiance_tx", &input_screen_radiance_tx_, no_interp);
  }
  if (raytracing.enabled()) {
    DRW_shgroup_uniform_block(grp, "hiz_buf", inst_.hiz.ubo_get());
    DRW_shgroup_uniform_texture_ref(grp, "hiz_tx", inst_.hiz_front.texture_ref_get());
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

void ForwardPass::render(const DRWView *view,
                         GPUTexture *depth_tx,
                         GPUTexture *UNUSED(combined_tx))
{
  HiZBuffer &hiz = inst_.hiz_front;

  DRW_stats_group_start("ForwardOpaque");

  DRW_draw_pass(prepass_ps_);
  hiz.set_dirty();

  if (inst_.raytracing.enabled()) {
    // rt_buffer.radiance_copy(combined_tx);
    hiz.update(depth_tx);
  }

  inst_.shadows.set_view(view, depth_tx);

  DRW_draw_pass(opaque_ps_);

  DRW_stats_group_end();

  DRW_stats_group_start("ForwardTransparent");
  /* TODO(fclem) This is suboptimal. We could sort during sync. */
  /* FIXME(fclem) This wont work for panoramic, where we need
   * to sort by distance to camera, not by z. */
  DRW_pass_sort_shgroup_z(transparent_ps_);
  DRW_draw_pass(transparent_ps_);
  DRW_stats_group_end();

  if (inst_.raytracing.enabled()) {
    // gbuffer.ray_radiance_tx.release();
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name DeferredPass
 * \{ */

void DeferredPass::sync(void)
{
  opaque_layer_.sync();
  refraction_layer_.sync();
  volumetric_layer_.sync();

  LightModule &lights = inst_.lights;
  LightProbeModule &lightprobes = inst_.lightprobes;
  GBuffer &gbuf = inst_.gbuffer;
  HiZBuffer &hiz = inst_.hiz_front;

  eGPUSamplerState no_interp = GPU_SAMPLER_DEFAULT;

  {
    DRWState state = DRW_STATE_WRITE_COLOR | DRW_STATE_STENCIL_NEQUAL | DRW_STATE_BLEND_ADD_FULL;
    eval_ps_ = DRW_pass_create("DeferredEval", state);
    GPUShader *sh = inst_.shaders.static_shader_get(DEFERRED_EVAL);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, eval_ps_);
    lights.shgroup_resources(grp);
    DRW_shgroup_uniform_block(grp, "sampling_buf", inst_.sampling.ubo_get());
    DRW_shgroup_uniform_block_ref(grp, "raytrace_buffer_buf", &ray_buffer_ubo_);
    DRW_shgroup_uniform_texture_ref(grp, "hiz_tx", hiz.texture_ref_get());
    DRW_shgroup_uniform_texture_ref(grp, "lightprobe_grid_tx", lightprobes.grid_tx_ref_get());
    DRW_shgroup_uniform_texture_ref(grp, "lightprobe_cube_tx", lightprobes.cube_tx_ref_get());
    DRW_shgroup_uniform_texture(grp, "utility_tx", inst_.shading_passes.utility_tx);
    DRW_shgroup_uniform_texture_ref(grp, "transmit_color_tx", &gbuf.transmit_color_tx);
    DRW_shgroup_uniform_texture_ref(grp, "transmit_normal_tx", &gbuf.transmit_normal_tx);
    DRW_shgroup_uniform_texture_ref(grp, "transmit_data_tx", &gbuf.transmit_data_tx);
    DRW_shgroup_uniform_texture_ref(grp, "reflect_color_tx", &gbuf.reflect_color_tx);
    DRW_shgroup_uniform_texture_ref(grp, "reflect_normal_tx", &gbuf.reflect_normal_tx);
    DRW_shgroup_uniform_texture_ref_ex(
        grp, "ray_radiance_diffuse_tx", &ray_radiance_diffuse_tx_, no_interp);
    DRW_shgroup_uniform_texture_ref_ex(
        grp, "ray_radiance_reflect_tx", &ray_radiance_reflect_tx_, no_interp);
    DRW_shgroup_uniform_texture_ref_ex(
        grp, "ray_radiance_refract_tx", &ray_radiance_refract_tx_, no_interp);
    DRW_shgroup_uniform_image_ref(grp, "rpass_diffuse_light", &gbuf.rpass_diffuse_light_tx);
    DRW_shgroup_uniform_image_ref(grp, "rpass_specular_light", &gbuf.rpass_specular_light_tx);
    // DRW_shgroup_uniform_image_ref(grp, "ray_data", &gbuf.rpass_specular_light_tx);
    DRW_shgroup_uniform_image_ref(grp, "sss_radiance", &gbuf.radiance_diffuse_tx);
    DRW_shgroup_uniform_texture_ref(
        grp, "sss_transmittance_tx", inst_.subsurface.transmittance_ref_get());
    DRW_shgroup_stencil_set(grp, 0x0, 0x0, CLOSURE_DIFFUSE | CLOSURE_REFLECTION);
    /* Sync with the Gbuffer pass. */
    DRW_shgroup_barrier(grp, GPU_BARRIER_TEXTURE_FETCH);
    DRW_shgroup_call_procedural_triangles(grp, nullptr, 1);
  }
  {
    DRWState state = DRW_STATE_WRITE_COLOR | DRW_STATE_STENCIL_EQUAL | DRW_STATE_BLEND_ADD_FULL;
    eval_subsurface_ps_ = DRW_pass_create("SubsurfaceEval", state);
    GPUShader *sh = inst_.shaders.static_shader_get(SUBSURFACE_EVAL);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, eval_subsurface_ps_);
    DRW_shgroup_uniform_block(grp, "sss_buf", inst_.subsurface.ubo_get());
    DRW_shgroup_uniform_block(grp, "hiz_buf", inst_.hiz.ubo_get());
    DRW_shgroup_uniform_texture_ref(grp, "hiz_tx", hiz.texture_ref_get());
    DRW_shgroup_uniform_texture_ref_ex(grp, "radiance_tx", &gbuf.radiance_diffuse_tx, no_interp);
    DRW_shgroup_uniform_texture_ref(grp, "transmit_color_tx", &gbuf.transmit_color_tx);
    DRW_shgroup_uniform_texture_ref(grp, "transmit_normal_tx", &gbuf.transmit_normal_tx);
    DRW_shgroup_uniform_texture_ref(grp, "transmit_data_tx", &gbuf.transmit_data_tx);
    DRW_shgroup_stencil_set(grp, 0x0, 0xFF, CLOSURE_SSS);
    /* Sync with the DeferredEval pass. */
    DRW_shgroup_barrier(grp, GPU_BARRIER_TEXTURE_FETCH);
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
                          RaytraceBuffer &rt_buffer_opaque_,
                          RaytraceBuffer &rt_buffer_refract_,
                          Texture &depth_tx,
                          GPUTexture *combined_tx)
{
  DRW_stats_group_start("OpaqueLayer");
  opaque_layer_.render(drw_view, &rt_buffer_opaque_, depth_tx, combined_tx);
  DRW_stats_group_end();

  rt_buffer_opaque_.render_end(drw_view);

  DRW_stats_group_start("RefractionLayer");
  refraction_layer_.render(drw_view, &rt_buffer_refract_, depth_tx, combined_tx);
  DRW_stats_group_end();

  rt_buffer_refract_.render_end(drw_view);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name DeferredLayer
 * \{ */

void DeferredLayer::deferred_shgroup_resources(DRWShadingGroup *grp)
{
  GBuffer &gbuf = inst_.gbuffer;
  /* Gbuffer. */
  DRW_shgroup_uniform_image_ref(grp, "gbuff_transmit_color", &gbuf.transmit_color_tx);
  DRW_shgroup_uniform_image_ref(grp, "gbuff_transmit_data", &gbuf.transmit_data_tx);
  DRW_shgroup_uniform_image_ref(grp, "gbuff_transmit_normal", &gbuf.transmit_normal_tx);
  DRW_shgroup_uniform_image_ref(grp, "gbuff_reflection_color", &gbuf.reflect_color_tx);
  DRW_shgroup_uniform_image_ref(grp, "gbuff_reflection_normal", &gbuf.reflect_normal_tx);
  DRW_shgroup_uniform_image_ref(grp, "gbuff_emission", &gbuf.emission_tx);
  /* Renderpasses. */
  DRW_shgroup_uniform_image_ref(grp, "rpass_volume_light", &gbuf.rpass_volume_light_tx);
}

void DeferredLayer::sync(void)
{
  closure_bits_ = eClosureBits(0);

  {
    DRWState state = DRW_STATE_WRITE_DEPTH | DRW_STATE_DEPTH_LESS;
    prepass_ps_ = DRW_pass_create("Gbuffer.Prepass", state);

    state |= DRW_STATE_CULL_BACK;
    prepass_culled_ps_ = DRW_pass_create("Gbuffer.Prepass.Culled", state);

    DRW_pass_link(prepass_ps_, prepass_culled_ps_);
  }
  {
    /* Only need one fragment to pass per pixel in order to use arbitrary load/store.
     * Use a combination of depth and stencil testing to achieve this. */
    DRWState state = DRW_STATE_WRITE_COLOR | DRW_STATE_DEPTH_EQUAL | DRW_STATE_STENCIL_NEQUAL |
                     DRW_STATE_WRITE_STENCIL | DRW_STATE_BLEND_CUSTOM;
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
  eClosureBits closure_bits = extract_closure_mask(gpumat);
  DRWPass *pass = (blender_mat->blend_flag & MA_BL_CULL_BACKFACE) ? gbuffer_culled_ps_ :
                                                                    gbuffer_ps_;
  DRWShadingGroup *grp = DRW_shgroup_material_create(gpumat, pass);
  DRW_shgroup_uniform_block(grp, "sampling_buf", inst_.sampling.ubo_get());
  DRW_shgroup_uniform_texture(grp, "utility_tx", inst_.shading_passes.utility_tx);
  deferred_shgroup_resources(grp);
  DRW_shgroup_stencil_set(grp, (uint)closure_bits & 0xFF, 0xFF, (uint)closure_bits & 0xFF);

  closure_bits_ |= closure_bits;
  return grp;
}

DRWShadingGroup *DeferredLayer::prepass_add(::Material *blender_mat, GPUMaterial *gpumat)
{
  DRWPass *pass = (blender_mat->blend_flag & MA_BL_CULL_BACKFACE) ? prepass_culled_ps_ :
                                                                    prepass_ps_;
  DRWShadingGroup *grp = DRW_shgroup_material_create(gpumat, pass);
  return grp;
}

void DeferredLayer::volume_add(Object *ob)
{
  (void)ob;
#if 0 /* TODO */
  LightModule &lights = inst_.lights;
  DeferredPass &deferred_pass = inst_.shading_passes.deferred;

  GPUShader *sh = inst_.shaders.static_shader_get(DEFERRED_VOLUME);
  DRWShadingGroup *grp = DRW_shgroup_create(sh, volume_ps_);
  lights.shgroup_resources(grp);
  DRW_shgroup_uniform_texture_ref(grp, "depth_max_tx", &deferred_pass.input_depth_behind_tx_);
  DRW_shgroup_uniform_texture(grp, "utility_tx", inst_.shading_passes.utility_tx);
  DRW_shgroup_stencil_set(grp, CLOSURE_VOLUME | CLOSURE_TRANSPARENCY, 0xFF, 0xFF);
  DRW_shgroup_call(grp, DRW_cache_cube_get(), ob);
#endif
}

void DeferredLayer::render(const DRWView *view,
                           RaytraceBuffer *rt_buffer,
                           Texture &depth_tx,
                           GPUTexture *UNUSED(combined_tx))
{
  DeferredPass &deferred_pass = inst_.shading_passes.deferred;
  GBuffer &gbuffer = inst_.gbuffer;
  HiZBuffer &hiz_front = inst_.hiz_front;
  HiZBuffer &hiz_back = inst_.hiz_back;

  const bool no_surfaces = DRW_pass_is_empty(gbuffer_ps_);
  const bool no_volumes = DRW_pass_is_empty(volume_ps_);
  if (no_surfaces && no_volumes) {
    return;
  }
  const bool use_subsurface = closure_bits_ & CLOSURE_SSS;
  const bool use_refraction = closure_bits_ & CLOSURE_REFRACTION;

  gbuffer.acquire(closure_bits_);

  if (use_refraction) {
    // rt_buffer->radiance_copy(combined_tx);
  }

  if (use_refraction) {
    hiz_back.update(depth_tx);
  }

  GPU_framebuffer_clear_stencil(GPU_framebuffer_active_get(), 0x00u);

  if (!no_surfaces) {
    DRW_draw_pass(prepass_ps_);
    hiz_front.set_dirty();
    hiz_back.set_dirty();
  }

  hiz_front.update(depth_tx);
  inst_.shadows.set_view(view, depth_tx);

  if (!no_surfaces) {
    DRW_draw_pass(gbuffer_ps_);
  }

  if (!no_volumes) {
    // DRW_draw_pass(volume_ps_);
  }

  if (!no_surfaces) {
    rt_buffer->trace(closure_bits_, depth_tx, deferred_pass);

    DRW_draw_pass(deferred_pass.eval_ps_);

    rt_buffer->release_tmp();

    if (use_subsurface) {
      DRW_draw_pass(deferred_pass.eval_subsurface_ps_);
    }
  }

  gbuffer.release();
}

/** \} */

}  // namespace blender::eevee