/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2021 Blender Foundation.
 */

/** \file
 * \ingroup eevee
 *
 * Module containing passes and parameters used for raytracing.
 * NOTE: For now only screen space raytracing is supported.
 */

#include <fstream>
#include <iostream>

#include "eevee_instance.hh"

#include "eevee_raytracing.hh"

namespace blender::eevee {

/* -------------------------------------------------------------------- */
/** \name Raytracing
 *
 * \{ */

void RaytracingModule::sync(void)
{
  SceneEEVEE &sce_eevee = inst_.scene->eevee;

  reflection_data.thickness = sce_eevee.ssr_thickness;
  reflection_data.brightness_clamp = (sce_eevee.ssr_firefly_fac < 1e-8f) ?
                                         FLT_MAX :
                                         sce_eevee.ssr_firefly_fac;
  reflection_data.quality = 1.0f - 0.95f * sce_eevee.ssr_quality;
  reflection_data.pool_offset = inst_.sampling.sample_get() / 5;

  refraction_data = static_cast<RaytraceData>(reflection_data);
  refraction_data.thickness = 1e16;
  /* TODO(fclem): Clamp option for refraction. */
  /* TODO(fclem): bias option for refraction. */
  /* TODO(fclem): bias option for refraction. */

  reflection_data.push_update();
  refraction_data.push_update();
  diffuse_data.push_update();

  // enabled_ = (sce_eevee.flag & SCE_EEVEE_RAYTRACING_ENABLED) != 0;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Raytracing Buffers
 *
 * \{ */

void RaytraceBuffer::sync(int2 extent)
{
  ray_data_diffuse_tx_.sync();
  ray_data_reflect_tx_.sync();
  ray_data_refract_tx_.sync();
  ray_radiance_diffuse_tx_.sync();
  ray_radiance_reflect_tx_.sync();
  ray_radiance_refract_tx_.sync();

  /* WORKAROUND(@fclem): Really stupid workaround to avoid the temp texture being the same
   * as the gbuffer ones. Change the extent by adding one pixel border. This is really bad
   * and we should rewrite the temp texture logic instead. */
  extent_ = extent + 1;
  if (false /* halfres */) {
    extent_.x = divide_ceil_u(extent_.x, 2);
    extent_.y = divide_ceil_u(extent_.y, 2);
    data_.res_scale = 2;
  }
  else {
    data_.res_scale = 1;
  }

  raygen_dispatch_size_.x = divide_ceil_u(extent_.x, RAYTRACE_GROUP_SIZE);
  raygen_dispatch_size_.y = divide_ceil_u(extent_.y, RAYTRACE_GROUP_SIZE);
  raygen_dispatch_size_.z = 1;

  GBuffer &gbuf = inst_.gbuffer;

  {
    /* Output rays and tile lists. */
    raygen_ps_ = DRW_pass_create("Raygen", (DRWState)0);
    GPUShader *sh = inst_.shaders.static_shader_get(RAYTRACE_RAYGEN);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, raygen_ps_);
    DRW_shgroup_storage_block(grp, "dispatch_diffuse_buf", dispatch_diffuse_buf_);
    DRW_shgroup_storage_block(grp, "dispatch_reflect_buf", dispatch_reflect_buf_);
    DRW_shgroup_storage_block(grp, "dispatch_refract_buf", dispatch_refract_buf_);
    DRW_shgroup_storage_block(grp, "tiles_diffuse_buf", tiles_diffuse_buf_);
    DRW_shgroup_storage_block(grp, "tiles_reflect_buf", tiles_reflect_buf_);
    DRW_shgroup_storage_block(grp, "tiles_refract_buf", tiles_refract_buf_);
    DRW_shgroup_uniform_texture_ref(grp, "gbuf_transmit_data_tx", &gbuf.transmit_data_tx);
    DRW_shgroup_uniform_texture_ref(grp, "gbuf_transmit_normal_tx", &gbuf.transmit_normal_tx);
    DRW_shgroup_uniform_texture_ref(grp, "gbuf_reflection_normal_tx", &gbuf.reflect_normal_tx);
    DRW_shgroup_uniform_texture_ref(grp, "depth_tx", &depth_view_tx_);
    DRW_shgroup_uniform_texture_ref(grp, "stencil_tx", &stencil_view_tx_);
    DRW_shgroup_uniform_image_ref(grp, "out_ray_data_diffuse", &ray_data_diffuse_tx_);
    DRW_shgroup_uniform_image_ref(grp, "out_ray_data_reflect", &ray_data_reflect_tx_);
    DRW_shgroup_uniform_image_ref(grp, "out_ray_data_refract", &ray_data_refract_tx_);
    DRW_shgroup_uniform_texture(grp, "utility_tx", inst_.shading_passes.utility_tx);
    DRW_shgroup_uniform_block(grp, "sampling_buf", inst_.sampling.ubo_get());
    DRW_shgroup_uniform_block(grp, "raytrace_buffer_buf", data_);
    DRW_shgroup_call_compute_ref(grp, raygen_dispatch_size_);
  }

  trace_diffuse_ps_ = sync_raytrace_passes("RayDiff",
                                           RAYTRACE_SCREEN_REFLECT,
                                           RAYTRACE_DENOISE_DIFFUSE,
                                           inst_.hiz_front,
                                           ray_data_diffuse_tx_,
                                           ray_radiance_diffuse_tx_,
                                           ray_history_diffuse_tx_[0],
                                           ray_history_diffuse_tx_[1],
                                           ray_variance_diffuse_tx_[0],
                                           ray_variance_diffuse_tx_[1],
                                           gbuf.transmit_data_tx,
                                           gbuf.transmit_normal_tx,
                                           inst_.raytracing.diffuse_data,
                                           dispatch_diffuse_buf_,
                                           tiles_diffuse_buf_);

  trace_reflect_ps_ = sync_raytrace_passes("RayRefl",
                                           RAYTRACE_SCREEN_REFLECT,
                                           RAYTRACE_DENOISE_REFLECT,
                                           inst_.hiz_front,
                                           ray_data_reflect_tx_,
                                           ray_radiance_reflect_tx_,
                                           ray_history_reflect_tx_[0],
                                           ray_history_reflect_tx_[1],
                                           ray_variance_reflect_tx_[0],
                                           ray_variance_reflect_tx_[1],
                                           gbuf.transmit_data_tx, /* UNUSED, Placeholder. */
                                           gbuf.reflect_normal_tx,
                                           inst_.raytracing.reflection_data,
                                           dispatch_reflect_buf_,
                                           tiles_reflect_buf_);

  trace_refract_ps_ = sync_raytrace_passes("RayRefr",
                                           RAYTRACE_SCREEN_REFRACT,
                                           RAYTRACE_DENOISE_REFRACT,
                                           inst_.hiz_back,
                                           ray_data_refract_tx_,
                                           ray_radiance_refract_tx_,
                                           ray_history_refract_tx_[0],
                                           ray_history_refract_tx_[1],
                                           ray_variance_refract_tx_[0],
                                           ray_variance_refract_tx_[1],
                                           gbuf.transmit_data_tx,
                                           gbuf.transmit_normal_tx,
                                           inst_.raytracing.refraction_data,
                                           dispatch_refract_buf_,
                                           tiles_refract_buf_);
}

DRWPass *RaytraceBuffer::sync_raytrace_passes(const char *name,
                                              eShaderType screen_trace_sh,
                                              eShaderType denoise_sh,
                                              HiZBuffer &hiz_tracing,
                                              TextureFromPool &ray_data_tx,
                                              TextureFromPool &ray_radiance_tx,
                                              Texture &ray_history_src_tx,
                                              Texture &ray_history_dst_tx,
                                              Texture &ray_variance_src_tx,
                                              Texture &ray_variance_dst_tx,
                                              TextureFromPool &gbuf_data_tx,
                                              TextureFromPool &gbuf_normal_tx,
                                              RaytraceDataBuf &ray_data_buf,
                                              RaytraceIndirectBuf &dispatch_buf,
                                              RaytraceTileBuf &tile_buf)
{
  LightProbeModule &lightprobes = inst_.lightprobes;

  DRWPass *pass = DRW_pass_create(name, (DRWState)0);
  /* Workaround when the tracing dispatch size can be higher than the max work group count in the
   * X dimension. */
  if ((raygen_dispatch_size_.x * raygen_dispatch_size_.y) > GPU_max_work_group_count(0)) {
    GPUShader *sh = inst_.shaders.static_shader_get(RAYTRACE_DISPATCH);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, pass);
    DRW_shgroup_storage_block(grp, "dispatch_buf", dispatch_buf);
    DRW_shgroup_barrier(grp, GPU_BARRIER_SHADER_STORAGE);
    DRW_shgroup_call_compute(grp, 1, 1, 1);
  }
  {
    GPUShader *sh = inst_.shaders.static_shader_get(screen_trace_sh);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, pass);
    DRW_shgroup_storage_block(grp, "dispatch_buf", dispatch_buf);
    DRW_shgroup_storage_block(grp, "tiles_buf", tile_buf);
    DRW_shgroup_uniform_block(grp, "hiz_buf", inst_.hiz.ubo_get());
    DRW_shgroup_uniform_block(grp, "raytrace_buffer_buf", data_);
    DRW_shgroup_uniform_block(grp, "raytrace_buf", ray_data_buf);
    DRW_shgroup_uniform_block(grp, "sampling_buf", inst_.sampling.ubo_get());
    DRW_shgroup_uniform_texture_ref(grp, "hiz_tx", hiz_tracing.texture_ref_get());
    DRW_shgroup_uniform_texture_ref(grp, "depth_tx", &depth_view_tx_);
    DRW_shgroup_uniform_image_ref(grp, "inout_ray_data", &ray_data_tx);
    DRW_shgroup_uniform_image_ref(grp, "out_ray_radiance", &ray_radiance_tx);
    DRW_shgroup_uniform_block(grp, "grids_buf", lightprobes.grid_ubo_get());
    DRW_shgroup_uniform_block(grp, "cubes_buf", lightprobes.cube_ubo_get());
    DRW_shgroup_uniform_block(grp, "probes_buf", lightprobes.info_ubo_get());
    DRW_shgroup_uniform_texture_ref(grp, "lightprobe_grid_tx", lightprobes.grid_tx_ref_get());
    DRW_shgroup_uniform_texture_ref(grp, "lightprobe_cube_tx", lightprobes.cube_tx_ref_get());
    DRW_shgroup_call_compute_indirect(grp, dispatch_buf);
    DRW_shgroup_barrier(grp, GPU_BARRIER_TEXTURE_FETCH);
  }
  {
    GPUShader *sh = inst_.shaders.static_shader_get(denoise_sh);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, pass);
    DRW_shgroup_storage_block(grp, "dispatch_buf", dispatch_buf);
    DRW_shgroup_storage_block(grp, "tiles_buf", tile_buf);
    DRW_shgroup_uniform_block(grp, "raytrace_buffer_buf", data_);
    DRW_shgroup_uniform_block(grp, "raytrace_buf", ray_data_buf);
    DRW_shgroup_uniform_texture_ref(grp, "gbuf_data_tx", &gbuf_data_tx);
    DRW_shgroup_uniform_texture_ref(grp, "gbuf_normal_tx", &gbuf_normal_tx);
    DRW_shgroup_uniform_texture_ref(grp, "depth_tx", &depth_view_tx_);
    DRW_shgroup_uniform_texture_ref(grp, "stencil_tx", &stencil_view_tx_);
    DRW_shgroup_uniform_texture_ref(grp, "ray_data_tx", &ray_data_tx);
    DRW_shgroup_uniform_texture_ref(grp, "ray_radiance_tx", &ray_radiance_tx);
    DRW_shgroup_uniform_texture_ref(grp, "ray_history_tx", &ray_history_src_tx);
    DRW_shgroup_uniform_texture_ref(grp, "ray_variance_tx", &ray_variance_src_tx);
    DRW_shgroup_uniform_image_ref(grp, "out_history_img", &ray_history_dst_tx);
    DRW_shgroup_uniform_image_ref(grp, "out_variance_img", &ray_variance_dst_tx);
    DRW_shgroup_call_compute_indirect(grp, dispatch_buf);
    DRW_shgroup_barrier(grp, GPU_BARRIER_TEXTURE_FETCH);
  }
  return pass;
}

void RaytraceBuffer::trace(eClosureBits closure_type,
                           Texture &depth_buffer,
                           DeferredPass &deferred_pass)
{
  GPU_storagebuf_clear_to_zero(dispatch_diffuse_buf_);
  GPU_storagebuf_clear_to_zero(dispatch_reflect_buf_);
  GPU_storagebuf_clear_to_zero(dispatch_refract_buf_);

  bool do_diffuse = bool(closure_type & CLOSURE_DIFFUSE);
  bool do_reflect = bool(closure_type & CLOSURE_REFLECTION);
  bool do_refract = bool(closure_type & CLOSURE_REFRACTION);
  ray_data_diffuse_tx_.acquire(do_diffuse ? extent_ : int2(1), GPU_RGBA16F, (void *)this);
  ray_data_reflect_tx_.acquire(do_reflect ? extent_ : int2(1), GPU_RGBA16F, (void *)this);
  ray_data_refract_tx_.acquire(do_refract ? extent_ : int2(1), GPU_RGBA16F, (void *)this);
  ray_radiance_diffuse_tx_.acquire(do_diffuse ? extent_ : int2(1), GPU_RGBA16F, (void *)this);
  ray_radiance_reflect_tx_.acquire(do_reflect ? extent_ : int2(1), GPU_RGBA16F, (void *)this);
  ray_radiance_refract_tx_.acquire(do_refract ? extent_ : int2(1), GPU_RGBA16F, (void *)this);

  data_.valid_history_diffuse = true;
  data_.valid_history_reflection = true;
  data_.valid_history_refraction = true;

  for (int i = 0; i < 2; i++) {
    if (ray_history_diffuse_tx_[i].ensure_2d(GPU_R11F_G11F_B10F, do_diffuse ? extent_ : int2(1))) {
      data_.valid_history_diffuse = false;
    }
    if (ray_history_reflect_tx_[i].ensure_2d(GPU_R11F_G11F_B10F, do_reflect ? extent_ : int2(1))) {
      data_.valid_history_reflection = false;
    }
    if (ray_history_refract_tx_[i].ensure_2d(GPU_R11F_G11F_B10F, do_refract ? extent_ : int2(1))) {
      data_.valid_history_refraction = false;
    }
    if (ray_variance_diffuse_tx_[i].ensure_2d(GPU_R8, do_diffuse ? extent_ : int2(1))) {
      data_.valid_history_diffuse = false;
    }
    if (ray_variance_reflect_tx_[i].ensure_2d(GPU_R8, do_reflect ? extent_ : int2(1))) {
      data_.valid_history_reflection = false;
    }
    if (ray_variance_refract_tx_[i].ensure_2d(GPU_R8, do_refract ? extent_ : int2(1))) {
      data_.valid_history_refraction = false;
    }
  }

  /* TODO(fclem): Rotate depending on sample. */
  data_.res_bias = int2(0);
  data_.push_update();

  depth_view_tx_ = depth_buffer;
  stencil_view_tx_ = depth_buffer.stencil_view();

  float4 clear_color(0.0f);

  DRW_draw_pass(raygen_ps_);
  if (do_diffuse) {
    GPU_texture_clear(ray_history_diffuse_tx_[1], GPU_DATA_FLOAT, &clear_color);
    GPU_texture_clear(ray_radiance_diffuse_tx_, GPU_DATA_FLOAT, &clear_color);
    DRW_draw_pass(trace_diffuse_ps_);
  }
  if (do_reflect) {
    GPU_texture_clear(ray_history_reflect_tx_[1], GPU_DATA_FLOAT, &clear_color);
    GPU_texture_clear(ray_radiance_reflect_tx_, GPU_DATA_FLOAT, &clear_color);
    DRW_draw_pass(trace_reflect_ps_);
  }
  if (do_refract) {
    GPU_texture_clear(ray_history_refract_tx_[1], GPU_DATA_FLOAT, &clear_color);
    GPU_texture_clear(ray_radiance_refract_tx_, GPU_DATA_FLOAT, &clear_color);
    DRW_draw_pass(trace_refract_ps_);
  }

  /* Pass deferred pass arguments. */
  deferred_pass.ray_buffer_ubo_ = data_;
  deferred_pass.ray_radiance_diffuse_tx_ = ray_history_diffuse_tx_[1];
  deferred_pass.ray_radiance_reflect_tx_ = ray_history_reflect_tx_[1];
  deferred_pass.ray_radiance_refract_tx_ = ray_history_refract_tx_[1];
}

void RaytraceBuffer::release_tmp()
{
  ray_data_diffuse_tx_.release();
  ray_data_reflect_tx_.release();
  ray_data_refract_tx_.release();
  ray_radiance_diffuse_tx_.release();
  ray_radiance_reflect_tx_.release();
  ray_radiance_refract_tx_.release();
}

/** \} */

}  // namespace blender::eevee
