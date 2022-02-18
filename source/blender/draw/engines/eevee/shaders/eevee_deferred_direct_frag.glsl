
/**
 * Direct lighting evaluation: Evaluate lights and light-probes contributions for all bsdfs.
 **/

#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(common_debug_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_culling_iter_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_gbuffer_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_sampling_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shadow_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_light_eval_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_lightprobe_eval_cubemap_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_lightprobe_eval_grid_lib.glsl)

void main(void)
{
  vec2 uv = uvcoordsvar.xy;
  float gbuffer_depth = texelFetch(hiz_tx, ivec2(gl_FragCoord.xy), 0).r;
  vec3 vP = get_view_space_from_depth(uv, gbuffer_depth);
  vec3 P = point_view_to_world(vP);
  vec3 V = cameraVec(P);

  vec4 tra_col_in = texture(transmit_color_tx, uv);
  vec4 tra_nor_in = texture(transmit_normal_tx, uv);
  vec4 tra_dat_in = texture(transmit_data_tx, uv);
  vec4 ref_col_in = texture(reflect_color_tx, uv);
  vec4 ref_nor_in = texture(reflect_normal_tx, uv);

  ClosureEmission emission = gbuffer_load_emission_data(emission_data_tx, uv);
  ClosureDiffuse diffuse = gbuffer_load_diffuse_data(tra_col_in, tra_nor_in, tra_dat_in);
  ClosureReflection reflection = gbuffer_load_reflection_data(ref_col_in, ref_nor_in);

  float thickness;
  gbuffer_load_global_data(tra_nor_in, thickness);

  float noise_offset = sampling_rng_1D_get(sampling_buf, SAMPLING_LIGHTPROBE);
  float noise = utility_tx_fetch(utility_tx, gl_FragCoord.xy, UTIL_BLUE_NOISE_LAYER).r;
  float random_probe = fract(noise + noise_offset);

  vec3 radiance_diffuse = vec3(0);
  vec3 radiance_reflection = vec3(0);
  vec3 R = -reflect(V, reflection.N);

  light_eval(diffuse, reflection, P, V, vP.z, thickness, radiance_diffuse, radiance_reflection);

  out_combined = vec4(emission.emission, 0.0);
  out_diffuse.rgb = radiance_diffuse;
  /* FIXME(fclem): This won't work after the first light batch since we use additive blending. */
  out_diffuse.a = fract(float(diffuse.sss_id) / 1024.0) * 1024.0;
  /* Do not apply color to diffuse term for SSS material. */
  if (diffuse.sss_id == 0u) {
    out_diffuse.rgb *= diffuse.color;
    out_combined.rgb += out_diffuse.rgb;
  }
  out_specular = radiance_reflection * reflection.color;
  out_combined.rgb += out_specular;
}
