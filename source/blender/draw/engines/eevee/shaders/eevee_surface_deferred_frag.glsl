
#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_surface_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_bsdf_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_gbuffer_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_nodetree_eval_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_sampling_lib.glsl)

void main(void)
{
  g_data = init_globals();

  float noise_offset = sampling_rng_1D_get(sampling_buf, SAMPLING_CLOSURE);
  float noise = utility_tx_fetch(utility_tx, gl_FragCoord.xy, UTIL_BLUE_NOISE_LAYER).r;
  g_data.closure_rand = fract(noise + noise_offset);
  /* TODO(fclem) other RNG. */
  g_data.transmit_rand = fract(g_data.closure_rand * 6.1803398875);

  float thickness = nodetree_thickness();

  nodetree_surface();

  float alpha = saturate(1.0 - avg(g_transparency_data.transmittance));
  vec3 V = cameraVec(g_data.P);

  if (alpha > 0.0) {
    g_diffuse_data.color /= alpha;
    g_reflection_data.color /= alpha;
    g_refraction_data.color /= alpha;
    g_emission_data.emission /= alpha;
  }

  if (gl_FrontFacing) {
    g_refraction_data.ior = safe_rcp(g_refraction_data.ior);
  }

  g_reflection_data.N = ensure_valid_reflection(g_data.Ng, V, g_reflection_data.N);

  {
    out_reflection_color = g_reflection_data.color;
    out_reflection_normal.xy = gbuffer_encode_normal(g_reflection_data.N);
    out_reflection_normal.z = max(1e-4, g_reflection_data.roughness);
  }

  if (g_data.transmit_rand == 0.0) {
    out_transmit_color = g_refraction_data.color;
    out_transmit_normal.xy = gbuffer_encode_normal(g_refraction_data.N);
    out_transmit_normal.z = -1.0;
    out_transmit_normal.w = thickness;
    out_transmit_data.x = g_refraction_data.ior;
    out_transmit_data.y = g_refraction_data.roughness;
  }
  else {
    if (g_diffuse_data.sss_id == 1u) {
      g_diffuse_data.sss_id = uint(resource_handle + 1);
    }
    /* Output diffuse / SSS in transmit data. */
    out_transmit_color = g_diffuse_data.color;
    out_transmit_normal.xy = gbuffer_encode_normal(g_diffuse_data.N);
    out_transmit_normal.z = fract(float(g_diffuse_data.sss_id) / 1024.0);
    out_transmit_normal.w = thickness;
    out_transmit_data = g_diffuse_data.sss_radius;
  }

  out_volume_data = gbuffer_store_volume_data(g_volume_data);
  out_emission_data = gbuffer_store_emission_data(g_emission_data);
  out_transparency_data = gbuffer_store_transparency_data(g_transparency_data);
}
