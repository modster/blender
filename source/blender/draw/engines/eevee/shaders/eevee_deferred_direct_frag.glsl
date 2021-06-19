
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
uniform usampler2D diffuse_data_tx;
uniform usampler2D reflection_data_tx;
uniform usampler2D lights_culling_tx;
uniform sampler2DArray utility_tx;
uniform sampler2DShadow shadow_atlas_tx;
uniform sampler2DArray lightprobe_grid_tx;
uniform samplerCubeArray lightprobe_cube_tx;

utility_tx_fetch_define(utility_tx);
utility_tx_sample_define(utility_tx);

in vec4 uvcoordsvar;

layout(location = 0) out vec4 out_combined;
layout(location = 1) out vec3 out_diffuse;
layout(location = 2) out vec3 out_specular;

/* Prototypes. */
void light_eval(ClosureDiffuse diffuse,
                ClosureReflection reflection,
                vec3 P,
                vec3 V,
                float vP_z,
                inout vec3 out_diffuse,
                inout vec3 out_specular);
vec3 lightprobe_grid_eval(ClosureDiffuse diffuse, vec3 P, float random_threshold);
vec3 lightprobe_cubemap_eval(ClosureReflection reflection, vec3 P, vec3 R, float random_threshold);

void main(void)
{
  float gbuffer_depth = texture(depth_tx, uvcoordsvar.xy).r;
  vec3 vP = get_view_space_from_depth(uvcoordsvar.xy, gbuffer_depth);
  vec3 P = point_view_to_world(vP);
  vec3 V = cameraVec(P);

  ClosureEmission emission = gbuffer_load_emission_data(emission_data_tx, uvcoordsvar.xy);
  ClosureDiffuse diffuse = gbuffer_load_diffuse_data(diffuse_data_tx, uvcoordsvar.xy);
  ClosureReflection reflection = gbuffer_load_reflection_data(reflection_data_tx, uvcoordsvar.xy);

  float noise_offset = sampling_rng_1D_get(sampling, SAMPLING_LIGHTPROBE);
  float noise = utility_tx_fetch(gl_FragCoord.xy, UTIL_BLUE_NOISE_LAYER).r;
  float random_probe = fract(noise + noise_offset);

  vec3 radiance_diffuse = vec3(0);
  vec3 radiance_reflection = vec3(0);
  vec3 R = -reflect(V, reflection.N);

  light_eval(diffuse, reflection, P, V, vP.z, radiance_diffuse, radiance_reflection);
  radiance_diffuse += lightprobe_grid_eval(diffuse, P, random_probe);
  radiance_reflection += lightprobe_cubemap_eval(reflection, P, R, random_probe);

  out_diffuse = radiance_diffuse * diffuse.color;
  out_specular = radiance_reflection * reflection.color;
  out_combined = vec4(out_diffuse + out_specular + emission.emission, 0.0);
}

#pragma BLENDER_REQUIRE_POST(eevee_light_eval_lib.glsl)
#pragma BLENDER_REQUIRE_POST(eevee_lightprobe_eval_cubemap_lib.glsl)
#pragma BLENDER_REQUIRE_POST(eevee_lightprobe_eval_grid_lib.glsl)
