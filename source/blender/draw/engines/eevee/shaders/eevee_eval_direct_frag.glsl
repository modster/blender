
/**
 * Direct lighting evaluation: Evaluate lights and light-probes contributions for all bsdfs.
 **/

#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_light_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_culling_iter_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shadow_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_gbuffer_lib.glsl)
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

uniform sampler2D depth_tx;
uniform usampler2D diffuse_data_tx;
uniform usampler2D reflection_data_tx;
uniform usampler2D lights_culling_tx;
uniform sampler2DArray utility_tx;
uniform sampler2DShadow shadow_atlas_tx;

in vec4 uvcoordsvar;

layout(location = 0) out vec3 out_combined;
layout(location = 1) out vec3 out_diffuse;
layout(location = 2) out vec3 out_glossy;

void main(void)
{
  float gbuffer_depth = texture(depth_tx, uvcoordsvar.xy).r;
  vec3 vP = get_view_space_from_depth(uvcoordsvar.xy, gbuffer_depth);
  vec3 P = point_view_to_world(vP);
  vec3 V = cameraVec(P);

  uvec4 diffuse_data = texture(diffuse_data_tx, uvcoordsvar.xy);
  GBufferDiffuseData diffuse = gbuffer_decode_diffuse_data(diffuse_data);

  uvec2 reflection_data = texture(reflection_data_tx, uvcoordsvar.xy).xy;
  GBufferReflectionData reflection = gbuffer_decode_reflection_data(reflection_data);

  vec2 uv = vec2(reflection.roughness, safe_sqrt(1.0 - dot(reflection.N, V)));
  uv = uv * UTIL_TEX_UV_SCALE + UTIL_TEX_UV_BIAS;
  vec4 ltc_mat = texture(utility_tx, vec3(uv, UTIL_LTC_MAT_LAYER));
  float ltc_mag = texture(utility_tx, vec3(uv, UTIL_LTC_MAG_LAYER)).x;

  vec3 radiance_diffuse = vec3(0);
  vec3 radiance_glossy = vec3(0);

  /* TODO(fclem) lightprobes */

  ITEM_FOREACH_BEGIN (light_culling, lights_culling_tx, vP.z, l_idx) {
    LightData light = lights[l_idx];
    vec3 L;
    float dist;
    light_vector_get(light, P, L, dist);

    float visibility = light_attenuation(light, L, dist);

    if (light.shadow_id != LIGHT_NO_SHADOW && (light.diffuse_power > 0.0 || visibility > 0.0)) {
      vec3 lL = light_world_to_local(light, -L) * dist;
      vec3 shadow_co = shadow_punctual_coordinates_get(shadows_punctual[l_idx], lL);
      visibility *= texture(shadow_atlas_tx, shadow_co);
    }

    if (visibility < 1e-6) {
      continue;
    }

    if (light.diffuse_power > 0.0) {
      float intensity = visibility * light.diffuse_power *
                        light_diffuse(utility_tx, light, diffuse.N, V, L, dist);
      radiance_diffuse += light.color * intensity;
    }

    if (light.specular_power > 0.0) {
      float intensity = visibility * light.specular_power *
                        light_ltc(utility_tx, light, reflection.N, V, L, dist, ltc_mat);
      radiance_glossy += light.color * intensity;
    }
  }
  ITEM_FOREACH_END

  out_diffuse = radiance_diffuse * diffuse.color;
  out_glossy = radiance_glossy * reflection.color;
  out_combined = out_diffuse + out_glossy;
}