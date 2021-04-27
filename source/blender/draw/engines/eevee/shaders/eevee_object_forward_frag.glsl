
#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_object_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_light_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_culling_iter_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shadow_lib.glsl)
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

layout(location = 0, index = 0) out vec4 outRadiance;
layout(location = 0, index = 1) out vec4 outTransmittance;

MeshData g_surf;
ivec4 g_closure_data[8];

void main(void)
{
  g_surf = init_from_interp();

  /* Prevent precision issues on unit coordinates. */
  vec3 p = (g_surf.P + 0.000001) * 0.999999;
  int xi = int(abs(floor(p.x)));
  int yi = int(abs(floor(p.y)));
  int zi = int(abs(floor(p.z)));
  bool check = ((mod(xi, 2) == mod(yi, 2)) == bool(mod(zi, 2)));

  float vP_z = get_view_z_from_depth(gl_FragCoord.z);

  vec3 radiance = vec3(0);
  ITEM_FOREACH_BEGIN (light_culling, lights_culling_tx, vP_z, l_idx) {
    LightData light = lights[l_idx];
    vec3 L;
    float dist;
    light_vector_get(light, g_surf.P, L, dist);

    float intensity = light_diffuse(utility_tx, light, g_surf.N, cameraVec(g_surf.P), L, dist) *
                      light.diffuse_power;

    float roughness = 0.25;
    float cos_theta = dot(g_surf.N, cameraVec(g_surf.P));
    vec2 uv = vec2(roughness, sqrt(1.0 - cos_theta));
    uv = uv * UTIL_TEX_UV_SCALE + UTIL_TEX_UV_BIAS;
    vec4 ltc_mat = texture(utility_tx, vec3(uv, UTIL_LTC_MAT_LAYER));
    float ltc_mag = texture(utility_tx, vec3(uv, UTIL_LTC_MAG_LAYER)).x;

    intensity += light_ltc(utility_tx, light, g_surf.N, cameraVec(g_surf.P), L, dist, ltc_mat) *
                 (light.specular_power * ltc_mag);

    if (light.shadow_id != LIGHT_NO_SHADOW && intensity > 0.0) {
      vec3 lL = light_world_to_local(light, -L) * dist;
      vec3 shadow_co = shadow_punctual_coordinates_get(shadows_punctual[l_idx], lL);
      intensity *= texture(shadow_atlas_tx, shadow_co);
    }

    radiance += light.color * intensity * light_attenuation(light, L, dist);
  }
  ITEM_FOREACH_END

  // outRadiance = vec4(radiance * mix(0.2, 0.8, check), 1.0);
  outRadiance = vec4(radiance, 1.0);
  outTransmittance = vec4(0.0, 0.0, 0.0, 1.0);
}