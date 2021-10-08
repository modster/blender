
/**
 * Direct lighting evaluation: Evaluate lights and light-probes contributions for all bsdfs.
 **/

#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_culling_iter_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_gbuffer_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_sampling_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)
#pragma BLENDER_REQUIRE(eevee_shadow_lib.glsl)

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

uniform sampler2D depth_tx;
uniform sampler2D emission_data_tx;
uniform sampler2D transmit_color_tx;
uniform sampler2D transmit_normal_tx;
uniform sampler2D transmit_data_tx;
uniform sampler2D reflect_color_tx;
uniform sampler2D reflect_normal_tx;
uniform sampler1D sss_transmittance_tx;
uniform usampler2D lights_culling_tx;
uniform sampler2DArray utility_tx;
uniform sampler2DShadow shadow_atlas_tx;
uniform sampler2D shadow_depth_tx;
uniform sampler2DArray lightprobe_grid_tx;
uniform samplerCubeArray lightprobe_cube_tx;

utility_tx_fetch_define(utility_tx);
utility_tx_sample_define(utility_tx);

in vec4 uvcoordsvar;

layout(location = 0) out vec4 out_combined;
layout(location = 1) out vec4 out_diffuse;
layout(location = 2) out vec3 out_specular;

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
  float gbuffer_depth = texture(depth_tx, uvcoordsvar.xy).r;
  vec3 vP = get_view_space_from_depth(uvcoordsvar.xy, gbuffer_depth);
  vec3 P = point_view_to_world(vP);
  vec3 V = cameraVec(P);

  vec4 tra_col_in = texture(transmit_color_tx, uvcoordsvar.xy);
  vec4 tra_nor_in = texture(transmit_normal_tx, uvcoordsvar.xy);
  vec4 tra_dat_in = texture(transmit_data_tx, uvcoordsvar.xy);
  vec4 ref_col_in = texture(reflect_color_tx, uvcoordsvar.xy);
  vec4 ref_nor_in = texture(reflect_normal_tx, uvcoordsvar.xy);

  ClosureEmission emission = gbuffer_load_emission_data(emission_data_tx, uvcoordsvar.xy);
  ClosureDiffuse diffuse = gbuffer_load_diffuse_data(tra_col_in, tra_nor_in, tra_dat_in);
  ClosureReflection reflection = gbuffer_load_reflection_data(ref_col_in, ref_nor_in);
  ClosureRefraction refraction = gbuffer_load_refraction_data(tra_col_in, tra_nor_in, tra_dat_in);

  float thickness;
  gbuffer_load_global_data(tra_nor_in, thickness);

  float noise_offset = sampling_rng_1D_get(sampling, SAMPLING_LIGHTPROBE);
  float noise = utility_tx_fetch(gl_FragCoord.xy, UTIL_BLUE_NOISE_LAYER).r;
  float random_probe = fract(noise + noise_offset);

  vec3 radiance_diffuse = vec3(0);
  vec3 radiance_reflection = vec3(0);
  vec3 radiance_refraction = vec3(0);
  vec3 R = -reflect(V, reflection.N);
  vec3 T = refract(-V, refraction.N, refraction.ior);

  light_eval(diffuse, reflection, P, V, vP.z, thickness, radiance_diffuse, radiance_reflection);
  radiance_diffuse += lightprobe_grid_eval(P, diffuse.N, random_probe);
  radiance_reflection += lightprobe_cubemap_eval(P, R, reflection.roughness, random_probe);
  radiance_refraction += lightprobe_cubemap_eval(P, T, sqr(refraction.roughness), random_probe);

  out_combined = vec4(emission.emission, 0.0);
  out_diffuse.rgb = radiance_diffuse;
  /* FIXME(fclem): This won't work after the first light batch since we use additive blending. */
  out_diffuse.a = fract(float(diffuse.sss_id) / 1024.0);
  /* Do not apply color to diffuse term for SSS material. */
  if (diffuse.sss_id == 0u) {
    out_diffuse.rgb *= diffuse.color;
    out_combined.rgb += out_diffuse.rgb;
  }
  out_specular = radiance_reflection * reflection.color + radiance_refraction * refraction.color;
  out_combined.rgb += out_specular;
}

#pragma BLENDER_REQUIRE_POST(eevee_light_eval_lib.glsl)
#pragma BLENDER_REQUIRE_POST(eevee_lightprobe_eval_cubemap_lib.glsl)
#pragma BLENDER_REQUIRE_POST(eevee_lightprobe_eval_grid_lib.glsl)
