/**
 * Generate ray direction from the gbuffer data.
 * Each ray type (reflection/refraction/diffuse) is put inside a different buffer.
 * Output is ray dir + ray pdf.
 * Each tile that contains any type of rays is written to a tile list.
 */

#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_gbuffer_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_raytrace_raygen_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_sampling_lib.glsl)

shared uint closures_bits;

void main()
{
  if (gl_LocalInvocationID.xy == uvec2(0)) {
    dispatch_diffuse_buf.num_groups_y = dispatch_diffuse_buf.num_groups_z = 1u;
    dispatch_reflect_buf.num_groups_y = dispatch_reflect_buf.num_groups_z = 1u;
    dispatch_refract_buf.num_groups_y = dispatch_refract_buf.num_groups_z = 1u;
  }

  if (gl_LocalInvocationID.xy == uvec2(0)) {
    closures_bits = 0u;
  }
  barrier();

  ivec2 texel = ivec2(gl_GlobalInvocationID.xy);
  ivec2 texel_fullres = texel * raytrace_buffer_buf.res_scale + raytrace_buffer_buf.res_bias;

  bool valid_texel = in_texture_range(texel_fullres, stencil_tx);
  uint local_closure_bits = (!valid_texel) ? 0u : texelFetch(stencil_tx, texel_fullres, 0).r;
  /* TODO(fclem): We could use wave ops instead of a shared variables. */
  atomicOr(closures_bits, local_closure_bits);
  barrier();

  /* Add the tile to the raytracing list. */
  if (gl_LocalInvocationID.xy == uvec2(0)) {
    uint tile_co = packUvec2x16(gl_WorkGroupID.xy);
    if (flag_test(closures_bits, CLOSURE_DIFFUSE)) {
      tiles_diffuse_buf[atomicAdd(dispatch_diffuse_buf.num_groups_x, 1u)] = tile_co;
    }
    if (flag_test(closures_bits, CLOSURE_REFLECTION)) {
      tiles_reflect_buf[atomicAdd(dispatch_reflect_buf.num_groups_x, 1u)] = tile_co;
    }
    if (flag_test(closures_bits, CLOSURE_REFRACTION)) {
      tiles_refract_buf[atomicAdd(dispatch_refract_buf.num_groups_x, 1u)] = tile_co;
    }
  }

  if (flag_test(closures_bits, CLOSURE_DIFFUSE | CLOSURE_REFRACTION | CLOSURE_REFLECTION)) {
    /* Generate ray. */
    vec2 uv = vec2(texel_fullres) / vec2(textureSize(depth_tx, 0).xy);
    float depth = 0.0;
    if (valid_texel) {
      depth = texelFetch(depth_tx, texel_fullres, 0).r;
    }
    vec3 P = get_world_space_from_depth(uv, depth);
    vec3 V = cameraVec(P);
    vec4 noise = utility_tx_fetch(utility_tx, texel, UTIL_BLUE_NOISE_LAYER).gbar;

    if (flag_test(local_closure_bits, CLOSURE_DIFFUSE | CLOSURE_REFRACTION)) {
      vec4 col_in = vec4(0.0); /* UNUSED */
      vec4 tra_nor_in = texelFetch(gbuf_transmit_normal_tx, texel_fullres, 0);
      vec4 tra_dat_in = texelFetch(gbuf_transmit_data_tx, texel_fullres, 0);

      bool is_refraction = (tra_nor_in.z == -1.0);
      if (is_refraction) {
        ClosureRefraction closure = gbuffer_load_refraction_data(col_in, tra_nor_in, tra_dat_in);
        vec4 ray;
        ray.xyz = raytrace_refraction_direction(sampling_buf, noise.xy, closure, V, ray.w);
        imageStore(out_ray_data_refract, texel, ray);
      }
      else {
        ClosureDiffuse closure = gbuffer_load_diffuse_data(col_in, tra_nor_in, tra_dat_in);
        vec4 ray;
        ray.xyz = raytrace_diffuse_direction(sampling_buf, noise.xy, closure, ray.w);
        imageStore(out_ray_data_diffuse, texel, ray);
      }
    }
    else {
      imageStore(out_ray_data_diffuse, texel, vec4(0.0));
      imageStore(out_ray_data_refract, texel, vec4(0.0));
    }

    if (flag_test(local_closure_bits, CLOSURE_REFLECTION)) {
      vec4 col_in = vec4(0.0); /* UNUSED */
      vec4 ref_nor_in = texelFetch(gbuf_reflection_normal_tx, texel_fullres, 0);

      ClosureReflection closure = gbuffer_load_reflection_data(col_in, ref_nor_in);
      vec4 ray;
      ray.xyz = raytrace_reflection_direction(sampling_buf, noise.xy, closure, V, ray.w);
      imageStore(out_ray_data_reflect, texel, ray);
    }
    else {
      imageStore(out_ray_data_reflect, texel, vec4(0.0));
    }
  }
}
