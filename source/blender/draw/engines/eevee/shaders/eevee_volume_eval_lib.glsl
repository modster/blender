
#pragma BLENDER_REQUIRE(eevee_nodetree_eval_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_sampling_lib.glsl)

struct Ray {
  vec3 origin;
  vec3 direction;
  float max_time;
};

vec3 volume_eval_light(vec3 P, float anisotropy)
{
  vec3 light_out = vec3(0.0);
  // LIGHT_ITER(g_volume)
  // {
  //   LightData ld = lights_data[i];

  //   if (ld.l_volume == 0.0) {
  //     continue;
  //   }

  //   vec3 L;
  //   float dist;
  //   light_vector_get(light, P, L, dist);

  //   float visibility = light_attenuation(light, L, dist);

  //   if (light.shadow_id != LIGHT_NO_SHADOW && visibility > 0.0) {
  //     vec3 lL = light_world_to_local(light, -L) * dist;
  //     vec3 shadow_co = shadow_punctual_coordinates_get(shadows[l_idx], lL);
  //     visibility *= texture(shadow_atlas_tx, shadow_co);
  //   }

  //   if (visibility > 1e-4) {
  //     float intensity = light_volume(utility_tx, light, V, L, dist, anisotropy);
  //     light_out += light.color * (intensity * visibility);
  //   }
  // }
  return light_out;
}

void volume_eval(vec3 P, vec3 V, float max_time, out vec3 out_radiance, out vec3 out_transmittance)
{
  // out_radiance = vec3(0);
  // out_transmittance = vec3(1);

  // VOLUME_ITER(g_volume)
  // {
  //   nodetree_eval();

  //   vec3 scattered_light = g_volume_data.emission +
  //                          g_volume_data.scattering * volume_eval_light(P,
  //                          g_volume_data.anisotropy);

  //   /* Emission does not work if there is no extinction because
  //    * Tr evaluates to 1.0 leading to Lscat = 0.0. (See T65771) */
  //   /* s_extinction. */
  //   g_volume_data.transmittance = max(vec3(1e-7) * step(1e-5, Lscat),
  //   g_volume_data.transmittance);
  //   /* Evaluate Scattering. */
  //   float s_len = abs(ray_len - prev_ray_len);
  //   prev_ray_len = ray_len;
  //   vec3 Tr = exp(-g_volume_data.transmittance * s_len);
  //   /* Integrate along the current step segment. */
  //   scattered_light = (scattered_light - scattered_light * Tr) *
  //                     safe_rcp(g_volume_data.transmittance);
  //   /* Accumulate and also take into account the transmittance from previous steps. */
  //   out_radiance += out_transmittance * scattered_light;
  //   out_transmittance *= Tr;

  //   if (all(lessThan(out_transmittance, vec3(1e-4)))) {
  //     break;
  //   }
  // }
}

/* Simple version that compute transmittance only. */
void volume_eval_homogenous(Ray ray, inout vec3 out_transmittance, out vec3 out_depth_time)
{
  // nodetree_eval();

  float step_len = length(ray.direction);

  out_transmittance *= exp(-g_volume_data.transmittance * step_len * ray.max_time);

  float threshold = interlieved_gradient_noise(gl_FragCoord.xy, 0, 0.0);
  /* Find depth at which we have threshold opacity. */
  out_depth_time = log(max(threshold, 1e-6)) * safe_rcp(-g_volume_data.transmittance * step_len);
  out_depth_time = min(out_depth_time, vec3(ray.max_time));
}
