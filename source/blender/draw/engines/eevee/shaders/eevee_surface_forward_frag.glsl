
/**
 * Forward lighting evaluation: Lighting is evaluated during the geometry rasterization.
 *
 * This is used by alpha blended materials and materials using Shader to RGB nodes.
 **/

#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_bsdf_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_culling_iter_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_light_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_lightprobe_eval_cubemap_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_lightprobe_eval_grid_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_nodetree_eval_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_sampling_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)
#pragma BLENDER_REQUIRE(eevee_shadow_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_surface_lib.glsl)

layout(std140) uniform sampling_block
{
  SamplingData sampling;
};

layout(std140) uniform lights_block
{
  LightData lights[CULLING_ITEM_BATCH];
};

layout(std140) uniform lights_culling_block
{
  CullingData light_culling;
};

layout(std140) uniform shadows_punctual_block
{
  ShadowPunctualData shadows_punctual[CULLING_ITEM_BATCH];
};

layout(std140) uniform grids_block
{
  GridData grids[GRID_MAX];
};

layout(std140) uniform cubes_block
{
  CubemapData cubes[CULLING_ITEM_BATCH];
};

layout(std140) uniform lightprobes_info_block
{
  LightProbeInfoData probes_info;
};

uniform usampler2D lights_culling_tx;
uniform sampler2DArray utility_tx;
uniform sampler2DShadow shadow_atlas_tx;
uniform sampler2DArray lightprobe_grid_tx;
uniform samplerCubeArray lightprobe_cube_tx;

utility_tx_fetch_define(utility_tx);
utility_tx_sample_define(utility_tx);

layout(location = 0, index = 0) out vec4 out_radiance;
layout(location = 0, index = 1) out vec4 out_transmittance;

void lightprobe_eval(ClosureDiffuse diffuse,
                     ClosureReflection reflection,
                     vec3 P,
                     vec3 V,
                     inout vec3 radiance_diffuse,
                     inout vec3 radiance_reflection)
{
  float noise_offset = sampling_rng_1D_get(sampling, SAMPLING_LIGHTPROBE);
  float noise = utility_tx_fetch(gl_FragCoord.xy, UTIL_BLUE_NOISE_LAYER).r;
  float random_probe = fract(noise + noise_offset);

  int grid_index;
  lightprobe_grid_select(probes_info.grids, grids, P, random_probe, grid_index);

  radiance_diffuse += lightprobe_grid_evaluate(
      probes_info.grids, lightprobe_grid_tx, grids[grid_index], P, g_diffuse_data.N);

  int cube_index;
  lightprobe_cubemap_select(probes_info.cubes, cubes, P, random_probe, cube_index);

  vec3 R = -reflect(V, reflection.N);
  radiance_reflection += lightprobe_cubemap_evaluate(
      probes_info.cubes, lightprobe_cube_tx, cubes[cube_index], P, R, reflection.roughness);
}

void main(void)
{
  g_data = init_globals();

  float noise_offset = sampling_rng_1D_get(sampling, SAMPLING_CLOSURE);
  float noise = utility_tx_fetch(gl_FragCoord.xy, UTIL_BLUE_NOISE_LAYER).r;
  g_data.closure_rand = fract(noise + noise_offset);

  nodetree_surface();

  float vP_z = get_view_z_from_depth(gl_FragCoord.z);
  vec3 V = cameraVec(g_data.P);
  vec3 P = g_data.P;

  vec2 uv = vec2(g_reflection_data.roughness, safe_sqrt(1.0 - dot(g_reflection_data.N, V)));
  uv = uv * UTIL_TEX_UV_SCALE + UTIL_TEX_UV_BIAS;
  vec4 ltc_mat = texture(utility_tx, vec3(uv, UTIL_LTC_MAT_LAYER));
  float ltc_mag = texture(utility_tx, vec3(uv, UTIL_LTC_MAG_LAYER)).x;

  vec3 radiance_diffuse = vec3(0);
  vec3 radiance_reflection = vec3(0);

  /* TODO(fclem) lightprobes */

  LIGHTS_EVAL(lights,
              shadow_atlas_tx,
              shadows_punctual,
              utility_tx,
              light_culling,
              lights_culling_tx,
              vP_z,
              P,
              V,
              g_diffuse_data,
              g_reflection_data,
              ltc_mat,
              radiance_diffuse,
              radiance_reflection);

  lightprobe_eval(g_diffuse_data, g_reflection_data, P, V, radiance_diffuse, radiance_reflection);

  // volume_eval(ray, volume_radiance, volume_transmittance, volume_depth);

  out_radiance.rgb = radiance_diffuse * g_diffuse_data.color;
  out_radiance.rgb += radiance_reflection * g_reflection_data.color;
  out_radiance.rgb += g_emission_data.emission;
  out_radiance.a = 0.0;

  out_transmittance.rgb = g_transparency_data.transmittance;
  out_transmittance.a = saturate(avg(out_transmittance.rgb));
}
