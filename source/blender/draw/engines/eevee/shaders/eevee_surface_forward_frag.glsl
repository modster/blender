
/**
 * Forward lighting evaluation: Lighting is evaluated during the geometry rasterization.
 *
 * This is used by alpha blended materials and materials using Shader to RGB nodes.
 **/

#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_bsdf_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_culling_iter_lib.glsl)
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
uniform sampler2D shadow_depth_tx;
uniform sampler1D sss_transmittance_tx;
uniform sampler2DArray lightprobe_grid_tx;
uniform samplerCubeArray lightprobe_cube_tx;

utility_tx_fetch_define(utility_tx);
utility_tx_sample_define(utility_tx);

layout(location = 0, index = 0) out vec4 out_radiance;
layout(location = 0, index = 1) out vec4 out_transmittance;

/* Prototypes. */
void light_eval(ClosureDiffuse diffuse,
                ClosureReflection reflection,
                vec3 P,
                vec3 V,
                float vP_z,
                float thickness,
                inout vec3 out_diffuse,
                inout vec3 out_specular);
vec3 lightprobe_grid_eval(vec3 P, vec3 N, float random_threshold);
vec3 lightprobe_cubemap_eval(vec3 P, vec3 R, float roughness, float random_threshold);

void main(void)
{
  g_data = init_globals();

  float noise = utility_tx_fetch(gl_FragCoord.xy, UTIL_BLUE_NOISE_LAYER).r;
  g_data.closure_rand = fract(noise + sampling_rng_1D_get(sampling, SAMPLING_CLOSURE));
  g_data.transmit_rand = -1.0;

  float thickness = nodetree_thickness();

  nodetree_surface();

  float vP_z = get_view_z_from_depth(gl_FragCoord.z);
  vec3 V = cameraVec(g_data.P);
  vec3 P = g_data.P;

  float noise_probe = utility_tx_fetch(gl_FragCoord.xy, UTIL_BLUE_NOISE_LAYER).g;
  float random_probe = fract(noise_probe + sampling_rng_1D_get(sampling, SAMPLING_LIGHTPROBE));

  if (gl_FrontFacing) {
    g_refraction_data.ior = safe_rcp(g_refraction_data.ior);
  }

  vec3 radiance_diffuse = vec3(0);
  vec3 radiance_reflection = vec3(0);
  vec3 radiance_refraction = vec3(0);
  vec3 R = -reflect(V, g_reflection_data.N);
  vec3 T = refract(-V, g_refraction_data.N, g_refraction_data.ior);

  light_eval(g_diffuse_data,
             g_reflection_data,
             P,
             V,
             vP_z,
             thickness,
             radiance_diffuse,
             radiance_reflection);
  radiance_diffuse += lightprobe_grid_eval(P, g_diffuse_data.N, random_probe);
  radiance_reflection += lightprobe_cubemap_eval(P, R, g_reflection_data.roughness, random_probe);
  radiance_refraction += lightprobe_cubemap_eval(
      P, T, sqr(g_refraction_data.roughness), random_probe);

  // volume_eval(ray, volume_radiance, volume_transmittance, volume_depth);

  out_radiance.rgb = radiance_diffuse * g_diffuse_data.color;
  out_radiance.rgb += radiance_reflection * g_reflection_data.color;
  out_radiance.rgb += radiance_refraction * g_refraction_data.color;
  out_radiance.rgb += g_emission_data.emission;
  out_radiance.a = 0.0;

  out_transmittance.rgb = g_transparency_data.transmittance;
  out_transmittance.a = saturate(avg(out_transmittance.rgb));
}

#pragma BLENDER_REQUIRE_POST(eevee_light_eval_lib.glsl)
#pragma BLENDER_REQUIRE_POST(eevee_lightprobe_eval_cubemap_lib.glsl)
#pragma BLENDER_REQUIRE_POST(eevee_lightprobe_eval_grid_lib.glsl)
