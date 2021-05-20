
#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_surface_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_light_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_culling_iter_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shadow_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_nodetree_eval_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)

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

uniform usampler2D lights_culling_tx;
uniform sampler2DArray utility_tx;
uniform sampler2DShadow shadow_atlas_tx;

layout(location = 0, index = 0) out vec4 out_radiance;
layout(location = 0, index = 1) out vec4 out_transmittance;

void main(void)
{
  g_data = init_globals();

  ntree_eval_set_defaults();

  nodetree_surface();

  float vP_z = get_view_z_from_depth(gl_FragCoord.z);
  vec3 V = cameraVec(g_data.P);

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
              g_data.P,
              V,
              g_diffuse_data,
              g_reflection_data,
              ltc_mat,
              radiance_diffuse,
              radiance_reflection);

  // volume_eval(ray, volume_radiance, volume_transmittance, volume_depth);

  out_radiance.rgb = radiance_diffuse + radiance_reflection;
  out_transmittance.rgb = vec3(0.0);
}