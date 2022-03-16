
#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_surface_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_bsdf_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_gbuffer_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_nodetree_eval_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_sampling_lib.glsl)

/* TODO(fclem): Renderpasses. */
#define output_renderpass(a, b, c, d)

void main(void)
{
  g_data = init_globals();

  float noise = utility_tx_fetch(utility_tx, gl_FragCoord.xy, UTIL_BLUE_NOISE_LAYER).r;
  float noise_offset = sampling_rng_1D_get(sampling_buf, SAMPLING_CLOSURE);
  g_data.closure_rand = fract(noise + noise_offset);
  g_data.transmit_rand = -1.0;

#if 0 /* TODO. */
  nodetree_volume();

  output_renderpass(rpass_volume_light, vec4(, 0.0));
#endif

  float thickness = nodetree_thickness();

  /* TODO(fclem) other RNG. */
  /* NOTE(fclem): This needs to be just before nodetree_surface in order to not be ovewritten by
   * other nodetree evaluations. */
  g_data.transmit_rand = fract(g_data.closure_rand * 6.1803398875);

  nodetree_surface();

  out_transmittance = vec4(1.0 - g_transparency_data.holdout);
  float transmittance_mono = saturate(avg(g_transparency_data.transmittance));
#if 1 /* TODO(fclem): Alpha clipped materials. */

  /* Apply transmittance. */
  out_transmittance *= vec4(g_transparency_data.transmittance, transmittance_mono);
#else
  /* Stochastique monochromatic transmittance.
   * Pixels are discarded based on alpha. We need to compensate the applied transmittance
   * term on all radiance channels. */
  if (transmittance_mono < 1.0) {
    float alpha = 1.0 - transmittance_mono;
    g_diffuse_data.color /= alpha;
    g_reflection_data.color /= alpha;
    g_refraction_data.color /= alpha;
    g_emission_data.emission /= alpha;
  }
#endif
  out_radiance = vec4(g_emission_data.emission, g_transparency_data.holdout);

  output_renderpass(rpass_emission, vec4(g_emission_data.emission, 0.0));

  if (gl_FrontFacing) {
    g_refraction_data.ior = safe_rcp(g_refraction_data.ior);
  }

  vec3 V = cameraVec(g_data.P);
  g_reflection_data.N = ensure_valid_reflection(g_data.Ng, V, g_reflection_data.N);

  ivec2 out_texel = ivec2(gl_FragCoord.xy);

  if (true) {
    imageStore(gbuff_emission, out_texel, vec4(g_emission_data.emission, 0.0));
  }

  if (true) {
    vec4 out_color;
    out_color.xyz = g_reflection_data.color;
    imageStore(gbuff_reflection_color, out_texel, out_color);

    vec4 out_normal;
    out_normal.xy = gbuffer_encode_normal(g_reflection_data.N);
    out_normal.z = max(1e-4, g_reflection_data.roughness);
    imageStore(gbuff_reflection_normal, out_texel, out_normal);
  }

  if (g_data.transmit_rand == 0.0) {
    vec4 out_color;
    out_color.xyz = g_refraction_data.color;
    imageStore(gbuff_transmit_color, out_texel, out_color);

    vec4 out_normal;
    out_normal.xy = gbuffer_encode_normal(g_refraction_data.N);
    out_normal.z = -1.0;
    out_normal.w = thickness;
    imageStore(gbuff_transmit_normal, out_texel, out_normal);

    vec4 out_data;
    out_data.x = g_refraction_data.ior;
    out_data.y = g_refraction_data.roughness;
    imageStore(gbuff_transmit_data, out_texel, out_data);
  }
  else {
    /* Output diffuse / SSS in transmit data. */
    vec4 out_color;
    out_color.xyz = g_diffuse_data.color;
    imageStore(gbuff_transmit_color, out_texel, out_color);

    vec4 out_normal;
    out_normal.xy = gbuffer_encode_normal(g_diffuse_data.N);
    out_normal.z = (g_diffuse_data.sss_id == 1u) ? fract(float(resource_handle + 1) / 1024.0) : 0;
    out_normal.w = thickness;
    imageStore(gbuff_transmit_normal, out_texel, out_normal);

    vec4 out_data;
    out_data.xyz = g_diffuse_data.sss_radius;
    imageStore(gbuff_transmit_data, out_texel, out_data);
  }
}
