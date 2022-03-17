
/**
 * Spatial ray reuse. Denoise raytrace result using ratio estimator.
 * Also add in temporal reuse.
 *
 * Input: Ray direction * hit time, Ray radiance, Ray hit depth
 * Ouput: Ray radiance reconstructed, Mean Ray hit depth, Radiance Variance
 *
 * Shader is specialized depending on the type of ray to denoise.
 *
 * Following "Stochastic All The Things: Raytracing in Hybrid Real-Time Rendering"
 * by Tomasz Stachowiak
 * https://www.ea.com/seed/news/seed-dd18-presentation-slides-raytracing
 */

#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_bsdf_microfacet_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_gbuffer_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_raytrace_resolve_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_raytrace_denoise_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_sampling_lib.glsl)

// #define USE_HISTORY

void history_weigh_and_accumulate(vec3 rgb_history,
                                  vec3 rgb_min,
                                  vec3 rgb_max,
                                  vec3 rgb_mean,
                                  vec3 rgb_deviation,
                                  inout vec3 rgb_accum,
                                  inout float weight_accum)
{
  /* Basically tells us how much the given sample is inside the range of values
   * we found during spatial reconstruction. */
  /* Slide 46. */
  vec3 dist = (rgb_history - rgb_mean) * safe_rcp(rgb_deviation);
  float weight = exp2(-10.0 * max_v3(dist));
  /* Slide 47. */
  rgb_history = clamp(rgb_history, rgb_min, rgb_max);

#ifdef USE_HISTORY
  rgb_accum += rgb_history * weight;
  weight_accum += weight;
#endif
}

void main(void)
{
  uint tile_id = gl_WorkGroupID.y * gl_WorkGroupSize.x + gl_WorkGroupID.x;
  /* The dispatch may bigger than necessary. */
  if (dispatch_buf.tile_count != 0u && tile_id >= dispatch_buf.tile_count) {
    return;
  }

  uvec2 tile = unpackUvec2x16(tiles_buf[tile_id]);
  ivec2 texel = ivec2(tile * gl_WorkGroupSize.xy + gl_LocalInvocationID.xy);
  ivec2 texel_fullres = texel * raytrace_buffer_buf.res_scale + raytrace_buffer_buf.res_bias;

  ivec2 img_size = textureSize(depth_tx, 0).xy;

  if (any(greaterThan(texel, img_size))) {
    return;
  }

  /* Skip pixels that have not been raytraced. */
  uint local_closure_bits = texelFetch(stencil_tx, texel_fullres, 0).r;
  if (!flag_test(local_closure_bits, CLOSURE_FLAG)) {
    imageStore(out_history_img, texel, vec4(0.0));
    imageStore(out_variance_img, texel, vec4(0.0));
    return;
  }

  float gbuffer_depth = texelFetch(depth_tx, texel_fullres, 0).r;
  vec2 uv = vec2(texel_fullres) * drw_view.viewport_size_inverse;
  vec3 P = get_world_space_from_depth(uv, gbuffer_depth);
  vec3 V = cameraVec(P);

  int sample_count = resolve_sample_max;
  vec4 col_in = vec4(0.0); /* UNUSED */
  vec4 nor_in = texelFetch(gbuf_normal_tx, texel_fullres, 0);

#if defined(DENOISE_DIFFUSE)
  vec4 dat_in = texelFetch(gbuf_data_tx, texel_fullres, 0);

  ClosureDiffuse diffuse = gbuffer_load_diffuse_data(col_in, nor_in, dat_in);

  if (diffuse.sss_radius.r < 0.0) {
    /* Refraction pixel. */
    imageStore(out_history_img, texel_fullres, vec4(0.0));
    imageStore(out_variance_img, texel_fullres, vec4(0.0));
    return;
  }

#  define BSDF_EVAL(R) bsdf_lambert(diffuse.N, R)
#  define VALID_HISTORY raytrace_buffer_buf.valid_history_diffuse

#elif defined(DENOISE_REFRACTION)
  vec4 dat_in = texelFetch(gbuf_data_tx, texel_fullres, 0);

  ClosureRefraction refraction = gbuffer_load_refraction_data(col_in, nor_in, dat_in);

  if (refraction.ior == -1.0) {
    /* Diffuse/SSS pixel. */
    imageStore(out_history_img, texel_fullres, vec4(0.0));
    imageStore(out_variance_img, texel_fullres, vec4(0.0));
    return;
  }

  float roughness_sqr = max(1e-3, sqr(refraction.roughness));

  /* TODO(fclem): Unfortunately Refraction ray reuse does not work great for some reasons.
   * To investigate. */
  if (roughness_sqr == 1e-3) {
    sample_count = 1;
  }

#  define BSDF_EVAL(R) btdf_ggx(refraction.N, R, V, roughness_sqr, refraction.ior)
#  define VALID_HISTORY raytrace_buffer_buf.valid_history_refraction

#elif defined(DENOISE_REFLECTION)
  ClosureReflection reflection = gbuffer_load_reflection_data(col_in, nor_in);

  float roughness_sqr = max(1e-3, sqr(reflection.roughness));

  if (roughness_sqr == 1e-3) {
    sample_count = 1;
  }

#  define BSDF_EVAL(R) bsdf_ggx(reflection.N, R, V, roughness_sqr)
#  define VALID_HISTORY raytrace_buffer_buf.valid_history_reflection

#endif

  /* ----- SPATIAL DENOISE ----- */

  /* Blue noise categorised into 4 sets of samples.
   * See "Stochastic all the things" presentation slide 32-37. */
  int sample_pool = int((gl_GlobalInvocationID.x & 1u) + (gl_GlobalInvocationID.y & 1u) * 2u);
  sample_pool = (sample_pool + raytrace_buf.pool_offset) % 4;
  int sample_id = sample_pool * resolve_sample_max;

  float hit_depth_mean = 0.0;
  vec3 rgb_mean = vec3(0.0);
  vec3 rgb_moment = vec3(0.0);
  vec3 radiance_accum = vec3(0.0);
  float weight_accum = 0.0;
  for (int i = 0; i < sample_count; i++, sample_id++) {
    ivec2 sample_texel = texel + resolve_sample_offsets[sample_id];

    vec4 ray_data = texelFetch(ray_data_tx, sample_texel, 0);
    vec4 ray_radiance = texelFetch(ray_radiance_tx, sample_texel, 0);

    vec3 R = normalize(ray_data.xyz);
    float ray_pdf_inv = ray_data.w;
    /* Skip invalid pixels. */
    if (ray_pdf_inv == 0.0) {
      continue;
    }

    /* Slide 54. */
    float weight = BSDF_EVAL(R) * ray_pdf_inv;

    radiance_accum += ray_radiance.rgb * weight;
    weight_accum += weight;

    hit_depth_mean += ray_radiance.a;
    rgb_mean += ray_radiance.rgb;
    rgb_moment += sqr(ray_radiance.rgb);
  }

  /* ----- TEMPORAL DENOISE ----- */

  /* Local statistics. */
  float sample_count_inv = 1.0 / float(sample_count);
  rgb_mean *= sample_count_inv;
  rgb_moment *= sample_count_inv;
  hit_depth_mean *= sample_count_inv;
  vec3 rgb_variance = abs(rgb_moment - sqr(rgb_mean));
  vec3 rgb_deviation = sqrt(rgb_variance);

  float variance = max_v3(rgb_variance);

  radiance_accum *= safe_rcp(weight_accum);
  weight_accum = 1.0;

  if (VALID_HISTORY) {
    vec3 rgb_min = rgb_mean - rgb_deviation;
    vec3 rgb_max = rgb_mean + rgb_deviation;

    /* Surface reprojection. */
    vec2 uv_surf = project_point(raytrace_buffer_buf.history_persmat, P).xy * 0.5 + 0.5;
    ivec2 texel_surf = ivec2(uv_surf * vec2(img_size) + 0.5);
    if (in_texture_range(texel_surf, ray_history_tx)) {
      vec3 radiance = texture(ray_history_tx, uv_surf).rgb;
      history_weigh_and_accumulate(
          radiance, rgb_min, rgb_max, rgb_mean, rgb_deviation, radiance_accum, weight_accum);

      /* Variance estimate (slide 41). */
      float variance_history = texture(ray_variance_tx, uv_surf).r;
      variance = mix(variance, variance_history, 0.5);
    }

#if CLOSURE_FLAG != CLOSURE_DIFFUSE
    /* Reflection reprojection. */
    vec3 P_hit = get_world_space_from_depth(uv, hit_depth_mean);
    vec2 uv_hit = project_point(raytrace_buffer_buf.history_persmat, P_hit).xy * 0.5 + 0.5;
    ivec2 texel_hit = ivec2(uv_hit * vec2(img_size) + 0.5);
    if (in_texture_range(texel_hit, ray_history_tx)) {
      vec3 radiance = texture(ray_history_tx, uv_hit).rgb;
      history_weigh_and_accumulate(
          radiance, rgb_min, rgb_max, rgb_mean, rgb_deviation, radiance_accum, weight_accum);
    }
#endif

    radiance_accum *= safe_rcp(weight_accum);
  }

  /* Save linear depth in alpha to speed-up the bilateral filter. */
  imageStore(out_history_img, texel, vec4(radiance_accum, dot(cameraForward, P)));
  imageStore(out_variance_img, texel, vec4(variance, 0.0, 0.0, 0.0));
}
