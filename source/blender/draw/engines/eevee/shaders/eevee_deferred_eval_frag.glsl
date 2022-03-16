
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
#pragma BLENDER_REQUIRE(eevee_raytrace_resolve_lib.glsl)

void main()
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
  ClosureRefraction refraction = gbuffer_load_refraction_data(tra_col_in, tra_nor_in, tra_dat_in);
  ClosureReflection reflection = gbuffer_load_reflection_data(ref_col_in, ref_nor_in);

  float thickness;
  gbuffer_load_global_data(tra_nor_in, thickness);

  vec3 radiance_diffuse = vec3(0);
  vec3 radiance_reflect = vec3(0);
  vec3 radiance_refract = vec3(0);

  light_eval(diffuse, reflection, P, V, vP.z, thickness, radiance_diffuse, radiance_reflect);

  if (tra_nor_in.z != -1.0) {
    radiance_diffuse += raytrace_resolve(
        texel, 3, 1.5, transmit_normal_tx, ray_data_diffuse_tx, ray_radiance_diffuse_tx);
  }
  else {
    radiance_refract += raytrace_resolve(
        texel, 1, 0.64, transmit_normal_tx, ray_data_refract_tx, ray_radiance_refract_tx);
  }
  if (true) {
    radiance_reflect += raytrace_resolve(
        texel, 1, 0.65, reflect_normal_tx, ray_data_reflect_tx, ray_radiance_reflect_tx);
  }

  out_combined = vec4(0.0);
  out_combined.xyz += radiance_reflect * reflection.color;
  out_combined.xyz += radiance_refract * refraction.color;

  // output_renderpass(rpass_specular_light, vec3(1.0), out_combined);
  // output_renderpass(rpass_diffuse_light, vec3(1.0), vec4(radiance_diffuse, 0.0));

  if (diffuse.sss_id != 0u) {
    imageStore(sss_radiance, texel, vec4(radiance_diffuse, float(diffuse.sss_id % 1024)));
  }
  else {
    out_combined.xyz += radiance_diffuse * diffuse.color;
  }
}
