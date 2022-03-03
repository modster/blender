
/**
 * Direct lighting evaluation: Evaluate lights and light-probes contributions for all bsdfs.
 **/

#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_culling_iter_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_gbuffer_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_sampling_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shadow_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_light_eval_lib.glsl)

void main(void)
{
  ivec2 texel = ivec2(gl_FragCoord.xy);
  float gbuffer_depth = texelFetch(hiz_tx, texel, 0).r;
  vec2 uv = uvcoordsvar.xy;
  vec3 vP = get_view_space_from_depth(uv, gbuffer_depth);
  vec3 P = point_view_to_world(vP);
  vec3 V = cameraVec(P);

  vec4 tra_col_in = texelFetch(transmit_color_tx, texel, 0);
  vec4 tra_nor_in = texelFetch(transmit_normal_tx, texel, 0);
  vec4 tra_dat_in = texelFetch(transmit_data_tx, texel, 0);
  vec4 ref_col_in = texelFetch(reflect_color_tx, texel, 0);
  vec4 ref_nor_in = texelFetch(reflect_normal_tx, texel, 0);

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

  out_combined = vec4(0.0);
  out_combined.xyz += radiance_reflection * reflection.color;

  if (diffuse.sss_id != 0u) {
    // imageStore(sss_radiance, texel, vec4(radiance_diffuse, float(diffuse.sss_id % 1024)));
  }
  else {
    out_combined.xyz += radiance_diffuse * diffuse.color;
  }

  // output_renderpass(rpass_diffuse_light, vec3(1.0), vec4(radiance_diffuse, 0.0));
  // output_renderpass(rpass_specular_light, vec3(1.0), vec4(radiance_reflection, 0.0));
}
