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

  reflection_data_.thickness = sce_eevee.ssr_thickness;
  reflection_data_.brightness_clamp = (sce_eevee.ssr_firefly_fac < 1e-8f) ?
                                          FLT_MAX :
                                          sce_eevee.ssr_firefly_fac;
  reflection_data_.max_roughness = sce_eevee.ssr_max_roughness + 0.01f;
  reflection_data_.quality = 1.0f - 0.95f * sce_eevee.ssr_quality;
  reflection_data_.bias = 0.8f + sce_eevee.ssr_quality * 0.15f;
  reflection_data_.pool_offset = inst_.sampling.sample_get() / 5;

  refraction_data_ = static_cast<RaytraceData>(reflection_data_);
  // refraction_data_.thickness = 1e16;
  /* TODO(fclem): Clamp option for refraction. */
  /* TODO(fclem): bias option for refraction. */
  /* TODO(fclem): bias option for refraction. */

  diffuse_data_ = static_cast<RaytraceData>(reflection_data_);
  diffuse_data_.max_roughness = 1.01f;

  reflection_data_.push_update();
  refraction_data_.push_update();
  diffuse_data_.push_update();

  enabled_ = (sce_eevee.flag & SCE_EEVEE_RAYTRACING_ENABLED) != 0;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Raytracing Buffers
 *
 * \{ */

void RaytraceBuffer::sync(ivec2 extent)
{
  extent_ = extent;
  dispatch_size_.x = divide_ceil_u(extent.x, 8);
  dispatch_size_.y = divide_ceil_u(extent.y, 8);
  dispatch_size_.z = 1;

  /* Make sure the history matrix is up to date. */
  data_.push_update();

  LightProbeModule &lightprobes = inst_.lightprobes;
  eGPUSamplerState no_interp = GPU_SAMPLER_DEFAULT;

  /* The raytracing buffer contains the draw passes since it is stored per view and we need to
   * dispatch compute shaders with the right workgroup size. */

  {
    DRWState state = DRW_STATE_WRITE_COLOR | DRW_STATE_STENCIL_NEQUAL;
    std::array<DRWShadingGroup *, 3> grps;
    bool do_rt = inst_.raytracing.enabled();
    {
      trace_reflection_ps_ = DRW_pass_create("TraceReflection", state);
      GPUShader *sh = inst_.shaders.static_shader_get(do_rt ? RAYTRACE_REFLECTION :
                                                              RAYTRACE_REFLECTION_FALLBACK);
      grps[0] = DRW_shgroup_create(sh, trace_reflection_ps_);
      DRW_shgroup_uniform_block(grps[0], "raytrace_block", inst_.raytracing.reflection_ubo_get());
      DRW_shgroup_stencil_set(grps[0], 0x0, 0x0, CLOSURE_REFLECTION);
    }
    {
      trace_refraction_ps_ = DRW_pass_create("TraceRefraction", state);
      GPUShader *sh = inst_.shaders.static_shader_get(do_rt ? RAYTRACE_REFRACTION :
                                                              RAYTRACE_REFRACTION_FALLBACK);
      grps[1] = DRW_shgroup_create(sh, trace_refraction_ps_);
      DRW_shgroup_uniform_block(grps[1], "raytrace_block", inst_.raytracing.refraction_ubo_get());
      DRW_shgroup_stencil_set(grps[1], 0x0, 0x0, CLOSURE_REFRACTION);
    }
    {
      trace_diffuse_ps_ = DRW_pass_create("TraceDiffuse", state);
      GPUShader *sh = inst_.shaders.static_shader_get(do_rt ? RAYTRACE_DIFFUSE :
                                                              RAYTRACE_DIFFUSE_FALLBACK);
      grps[2] = DRW_shgroup_create(sh, trace_diffuse_ps_);
      DRW_shgroup_uniform_block(grps[2], "raytrace_block", inst_.raytracing.diffuse_ubo_get());
      DRW_shgroup_stencil_set(grps[2], 0x0, 0x0, CLOSURE_DIFFUSE);
    }

    for (DRWShadingGroup *grp : grps) {
      DRW_shgroup_uniform_block(grp, "sampling_block", inst_.sampling.ubo_get());
      DRW_shgroup_uniform_block(grp, "hiz_block", inst_.hiz.ubo_get());
      DRW_shgroup_uniform_block(grp, "cubes_block", lightprobes.cube_ubo_get());
      DRW_shgroup_uniform_block(grp, "lightprobes_info_block", lightprobes.info_ubo_get());
      DRW_shgroup_uniform_texture_ref(grp, "hiz_tx", &input_hiz_tx_);
      DRW_shgroup_uniform_texture_ref(grp, "hiz_front_tx", &input_hiz_front_tx_);
      DRW_shgroup_uniform_texture_ref(grp, "lightprobe_cube_tx", lightprobes.cube_tx_ref_get());
      DRW_shgroup_uniform_texture_ref_ex(grp, "radiance_tx", &input_radiance_tx_, no_interp);
      DRW_shgroup_uniform_texture_ref_ex(grp, "combined_tx", &input_combined_tx_, no_interp);
      DRW_shgroup_uniform_texture_ref_ex(grp, "cl_color_tx", &input_cl_color_tx_, no_interp);
      DRW_shgroup_uniform_texture_ref_ex(grp, "cl_normal_tx", &input_cl_normal_tx_, no_interp);
      DRW_shgroup_uniform_texture_ref_ex(grp, "cl_data_tx", &input_cl_data_tx_, no_interp);
      DRW_shgroup_uniform_texture(grp, "utility_tx", inst_.shading_passes.utility_tx);
      DRW_shgroup_call_procedural_triangles(grp, nullptr, 1);
    }
  }

  {
    /* Compute stage. No state needed. */
    DRWState state = (DRWState)0;
    std::array<DRWShadingGroup *, 3> grps;
    {
      denoise_reflection_ps_ = DRW_pass_create("DenoiseReflection", state);
      GPUShader *sh = inst_.shaders.static_shader_get(RAYTRACE_DENOISE_REFLECTION);
      grps[0] = DRW_shgroup_create(sh, denoise_reflection_ps_);
    }
    {
      denoise_refraction_ps_ = DRW_pass_create("DenoiseRefraction", state);
      GPUShader *sh = inst_.shaders.static_shader_get(RAYTRACE_DENOISE_REFRACTION);
      grps[1] = DRW_shgroup_create(sh, denoise_refraction_ps_);
    }
    {
      denoise_diffuse_ps_ = DRW_pass_create("DenoiseDiffuse", state);
      GPUShader *sh = inst_.shaders.static_shader_get(RAYTRACE_DENOISE_DIFFUSE);
      grps[2] = DRW_shgroup_create(sh, denoise_diffuse_ps_);
    }

    for (DRWShadingGroup *grp : grps) {
      /* Does not matter which raytrace_block we use. */
      DRW_shgroup_uniform_block(grp, "raytrace_block", inst_.raytracing.diffuse_ubo_get());
      DRW_shgroup_uniform_block(grp, "hiz_block", inst_.hiz.ubo_get());
      DRW_shgroup_uniform_block(grp, "rtbuffer_block", data_.ubo_get());
      DRW_shgroup_uniform_texture_ref_ex(grp, "ray_data_tx", &input_ray_data_tx_, no_interp);
      DRW_shgroup_uniform_texture_ref_ex(grp, "ray_radiance_tx", &input_ray_color_tx_, no_interp);
      DRW_shgroup_uniform_texture_ref_ex(grp, "hiz_tx", &input_hiz_front_tx_, no_interp);
      DRW_shgroup_uniform_texture_ref_ex(grp, "cl_color_tx", &input_cl_color_tx_, no_interp);
      DRW_shgroup_uniform_texture_ref_ex(grp, "cl_normal_tx", &input_cl_normal_tx_, no_interp);
      DRW_shgroup_uniform_texture_ref_ex(grp, "cl_data_tx", &input_cl_data_tx_, no_interp);
      DRW_shgroup_uniform_texture_ref(grp, "ray_history_tx", &input_history_tx_);
      DRW_shgroup_uniform_texture_ref(grp, "ray_variance_tx", &input_variance_tx_);
      DRW_shgroup_uniform_image_ref(grp, "out_history_img", &output_history_tx_);
      DRW_shgroup_uniform_image_ref(grp, "out_variance_img", &output_variance_tx_);
      DRW_shgroup_call_compute_ref(grp, dispatch_size_);
      DRW_shgroup_barrier(grp, GPU_BARRIER_SHADER_IMAGE_ACCESS);
    }
  }

  {
    DRWState state = DRW_STATE_WRITE_COLOR | DRW_STATE_STENCIL_NEQUAL | DRW_STATE_BLEND_ADD_FULL;
    std::array<DRWShadingGroup *, 3> grps;
    {
      resolve_reflection_ps_ = DRW_pass_create("ResolveReflection", state);
      GPUShader *sh = inst_.shaders.static_shader_get(RAYTRACE_RESOLVE_REFLECTION);
      grps[0] = DRW_shgroup_create(sh, resolve_reflection_ps_);
      DRW_shgroup_stencil_set(grps[0], 0x0, 0x0, CLOSURE_REFLECTION);
    }
    {
      resolve_refraction_ps_ = DRW_pass_create("ResolveRefraction", state);
      GPUShader *sh = inst_.shaders.static_shader_get(RAYTRACE_RESOLVE_REFRACTION);
      grps[1] = DRW_shgroup_create(sh, resolve_refraction_ps_);
      DRW_shgroup_stencil_set(grps[1], 0x0, 0x0, CLOSURE_REFRACTION);
    }
    {
      resolve_diffuse_ps_ = DRW_pass_create("ResolveDiffuse", state);
      GPUShader *sh = inst_.shaders.static_shader_get(RAYTRACE_RESOLVE_DIFFUSE);
      grps[2] = DRW_shgroup_create(sh, resolve_diffuse_ps_);
      DRW_shgroup_stencil_set(grps[2], 0x0, 0x0, CLOSURE_DIFFUSE);
    }

    for (DRWShadingGroup *grp : grps) {
      DRW_shgroup_uniform_block(grp, "hiz_block", inst_.hiz.ubo_get());
      DRW_shgroup_uniform_texture_ref_ex(grp, "ray_radiance_tx", &output_history_tx_, no_interp);
      DRW_shgroup_uniform_texture_ref_ex(grp, "ray_variance_tx", &output_variance_tx_, no_interp);
      DRW_shgroup_uniform_texture_ref_ex(grp, "cl_color_tx", &input_cl_color_tx_, no_interp);
      DRW_shgroup_uniform_texture_ref_ex(grp, "cl_normal_tx", &input_cl_normal_tx_, no_interp);
      DRW_shgroup_uniform_texture_ref_ex(grp, "cl_data_tx", &input_cl_data_tx_, no_interp);
      // DRW_shgroup_call_compute_ref(grp, dispatch_size_);
      // DRW_shgroup_barrier(grp, GPU_BARRIER_SHADER_IMAGE_ACCESS);
      DRW_shgroup_call_procedural_triangles(grp, nullptr, 1);
    }
  }
}

void RaytraceBuffer::trace(eClosureBits closure_type,
                           GBuffer &gbuffer,
                           HiZBuffer &hiz,
                           HiZBuffer &hiz_front)
{
  gbuffer.bind_tracing();

  input_hiz_tx_ = hiz.texture_get();
  input_hiz_front_tx_ = hiz_front.texture_get();
  if (closure_type == CLOSURE_REFLECTION) {
    input_cl_color_tx_ = gbuffer.reflect_color_tx;
    input_cl_normal_tx_ = gbuffer.reflect_normal_tx;
    input_cl_data_tx_ = gbuffer.reflect_normal_tx;
  }
  else {
    input_cl_color_tx_ = gbuffer.transmit_color_tx;
    input_cl_normal_tx_ = gbuffer.transmit_normal_tx;
    input_cl_data_tx_ = gbuffer.transmit_data_tx;
  }

  switch (closure_type) {
    default:
    case CLOSURE_REFLECTION:
      input_radiance_tx_ = gbuffer.combined_tx;
      DRW_draw_pass(trace_reflection_ps_);
      break;
    case CLOSURE_REFRACTION:
      input_radiance_tx_ = gbuffer.combined_tx;
      DRW_draw_pass(trace_refraction_ps_);
      break;
    case CLOSURE_DIFFUSE:
      input_radiance_tx_ = gbuffer.diffuse_tx;
      input_combined_tx_ = gbuffer.combined_tx;
      DRW_draw_pass(trace_diffuse_ps_);
      break;
  }

  input_ray_data_tx_ = gbuffer.ray_data_tx;
  input_ray_color_tx_ = gbuffer.ray_radiance_tx;
}

void RaytraceBuffer::denoise(eClosureBits closure_type)
{
  switch (closure_type) {
    default:
    case CLOSURE_REFLECTION:
      input_history_tx_ = reflection_radiance_history_get();
      input_variance_tx_ = reflection_variance_history_get();
      output_history_tx_ = reflection_radiance_get();
      output_variance_tx_ = reflection_variance_get();
      DRW_draw_pass(denoise_reflection_ps_);
      break;
    case CLOSURE_REFRACTION:
      input_history_tx_ = refraction_radiance_history_get();
      input_variance_tx_ = refraction_variance_history_get();
      output_history_tx_ = refraction_radiance_get();
      output_variance_tx_ = refraction_variance_get();
      DRW_draw_pass(denoise_refraction_ps_);
      break;
    case CLOSURE_DIFFUSE:
      input_history_tx_ = diffuse_radiance_history_get();
      input_variance_tx_ = diffuse_variance_history_get();
      output_history_tx_ = diffuse_radiance_get();
      output_variance_tx_ = diffuse_variance_get();
      DRW_draw_pass(denoise_diffuse_ps_);
      break;
  }
}

void RaytraceBuffer::resolve(eClosureBits closure_type, GBuffer &gbuffer)
{
  gbuffer.bind_radiance();
  switch (closure_type) {
    default:
    case CLOSURE_REFLECTION:
      DRW_draw_pass(resolve_reflection_ps_);
      break;
    case CLOSURE_REFRACTION:
      DRW_draw_pass(resolve_refraction_ps_);
      break;
    case CLOSURE_DIFFUSE:
      DRW_draw_pass(resolve_diffuse_ps_);
      break;
  }
}

/** \} */

}  // namespace blender::eevee
