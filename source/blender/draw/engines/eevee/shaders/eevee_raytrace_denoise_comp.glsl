
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
#pragma BLENDER_REQUIRE(eevee_sampling_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)

layout(local_size_x = 8, local_size_y = 8) in;

layout(std140) uniform raytrace_block
{
  RaytraceData raytrace;
};

layout(std140) uniform rtbuffer_block
{
  RaytraceBufferData rtbuffer;
};

layout(std140) uniform hiz_block
{
  HiZData hiz;
};

uniform sampler2D hiz_tx;
uniform sampler2D ray_data_tx;
uniform sampler2D ray_radiance_tx;
uniform sampler2D cl_color_tx;
uniform sampler2D cl_normal_tx;
uniform sampler2D cl_data_tx;
uniform sampler2D ray_history_tx;
uniform sampler2D ray_variance_tx;

layout(rgba16f) restrict uniform image2D out_history_img;
layout(r8) restrict uniform image2D out_variance_img;

//#define USE_HISTORY

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
  ivec2 texel = ivec2(gl_GlobalInvocationID.xy);
  ivec2 img_size = textureSize(ray_data_tx, 0).xy;

  if (any(greaterThan(texel, img_size))) {
    return;
  }

  /* Skip pixels that have not been raytraced. */
  if (texelFetch(ray_data_tx, texel, 0).w == 0.0) {
    imageStore(out_history_img, texel, vec4(0.0));
    imageStore(out_variance_img, texel, vec4(0.0));
    return;
  }

  vec2 uv = vec2(texel) / vec2(img_size);
  float gbuffer_depth = texelFetch(hiz_tx, texel, 0).r;
  vec3 vP = get_view_space_from_depth(uv, gbuffer_depth);
  vec3 vV = viewCameraVec(vP);
  vec2 texel_size = hiz.pixel_to_ndc * 0.5;

  int sample_count = resolve_sample_max;
#if defined(DIFFUSE)
  vec4 tra_col_in = texture(cl_color_tx, uv);
  vec4 tra_nor_in = texture(cl_normal_tx, uv);
  vec4 tra_dat_in = vec4(0.0); /* UNUSED */

  ClosureDiffuse diffuse = gbuffer_load_diffuse_data(tra_col_in, tra_nor_in, tra_dat_in);

  if (diffuse.sss_radius.r < 0.0) {
    /* Refraction pixel. */
    imageStore(out_history_img, texel, vec4(0.0));
    imageStore(out_variance_img, texel, vec4(0.0));
    return;
  }

  vec3 vN = transform_direction(ViewMatrix, diffuse.N);
  vec3 color = diffuse.color;

#elif defined(REFRACTION)
  vec4 tra_col_in = texture(cl_color_tx, uv);
  vec4 tra_nor_in = texture(cl_normal_tx, uv);
  vec4 tra_dat_in = texture(cl_data_tx, uv);

  ClosureRefraction refraction = gbuffer_load_refraction_data(tra_col_in, tra_nor_in, tra_dat_in);

  if (refraction.ior == -1.0) {
    /* Diffuse/SSS pixel. */
    imageStore(out_history_img, texel, vec4(0.0));
    imageStore(out_variance_img, texel, vec4(0.0));
    return;
  }
  float thickness;
  gbuffer_load_global_data(tra_nor_in, thickness);

  vec3 vN = transform_direction(ViewMatrix, refraction.N);
  float roughness_sqr = max(1e-3, sqr(refraction.roughness));
  vec3 color = refraction.color;

  /* TODO(fclem): Unfortunately Refraction ray reuse does not work great for some reasons.
   * To investigate. */
  sample_count = 1;
#else
  ClosureReflection reflection = gbuffer_load_reflection_data(cl_color_tx, cl_normal_tx, uv);

  vec3 vN = transform_direction(ViewMatrix, reflection.N);
  float roughness_sqr = max(1e-3, sqr(reflection.roughness));
  vec3 color = reflection.color;

  if (roughness_sqr == 1e-3) {
    sample_count = 1;
  }
#endif

  /* ----- SPATIAL DENOISE ----- */

  /* Blue noise categorised into 4 sets of samples.
   * See "Stochastic all the things" presentation slide 32-37. */
  int sample_pool = int((gl_GlobalInvocationID.x & 1u) + (gl_GlobalInvocationID.y & 1u) * 2u);
  sample_pool = (sample_pool + raytrace.pool_offset) % 4;
  int sample_id = sample_pool * resolve_sample_max;

  float hit_depth_mean = 0.0;
  vec3 rgb_mean = vec3(0.0);
  vec3 rgb_moment = vec3(0.0);
  vec3 radiance_accum = vec3(0.0);
  float weight_accum = 0.0;
  for (int i = 0; i < sample_count; i++, sample_id++) {
    vec2 sample_uv = uv + resolve_sample_offsets[sample_id] * texel_size;

    vec4 ray_data = texture(ray_data_tx, sample_uv);
    vec4 ray_radiance = texture(ray_radiance_tx, sample_uv);

    vec3 vR = normalize(ray_data.xyz);
    float ray_pdf_inv = ray_data.w;
    /* Skip invalid pixels. */
    if (ray_pdf_inv == 0.0) {
      continue;
    }

    /* Slide 54. */
#if defined(DIFFUSE)
    float bsdf = saturate(dot(vN, vR));
#elif defined(REFRACTION)
    float bsdf = btdf_ggx(vN, vR, vV, roughness_sqr, refraction.ior);
#else
    float bsdf = bsdf_ggx(vN, vR, vV, roughness_sqr);
#endif
    float weight = bsdf * ray_pdf_inv;

#ifdef REFRACTION
    /* Transmit twice if thickness is set and ray is longer than thickness. */
    if (thickness > 0.0 && length(ray_data.xyz) > thickness) {
      ray_radiance.rgb *= color;
    }
#endif
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

#if defined(DIFFUSE)
  if (rtbuffer.valid_history_diffuse) {
#elif defined(REFRACTION)
  if (rtbuffer.valid_history_refraction) {
#else
  if (rtbuffer.valid_history_reflection) {
#endif

    vec3 rgb_min = rgb_mean - rgb_deviation;
    vec3 rgb_max = rgb_mean + rgb_deviation;

    /* Surface reprojection. */
    vec3 P_surf = transform_point(ViewMatrixInverse, vP);
    vec2 uv_surf = project_point(rtbuffer.history_persmat, P_surf).xy * 0.5 + 0.5;
    ivec2 texel_surf = ivec2(uv_surf * vec2(img_size));
    if (all(lessThan(texel_surf, img_size)) && all(greaterThan(texel_surf, ivec2(0)))) {
      vec3 radiance = texelFetch(ray_history_tx, texel_surf, 0).rgb;
      history_weigh_and_accumulate(
          radiance, rgb_min, rgb_max, rgb_mean, rgb_deviation, radiance_accum, weight_accum);

      /* Variance estimate (slide 41). */
      float variance_history = texelFetch(ray_variance_tx, texel_surf, 0).r;
      variance = mix(variance, variance_history, 0.5);
    }

#if !defined(DIFFUSE)
    /* Reflexion reprojection. */
    vec3 P_hit = get_world_space_from_depth(uv, hit_depth_mean);
    vec2 uv_hit = project_point(rtbuffer.history_persmat, P_hit).xy * 0.5 + 0.5;
    ivec2 texel_hit = ivec2(uv_hit * vec2(img_size));
    if (all(lessThan(texel_hit, img_size)) && all(greaterThan(texel_hit, ivec2(0)))) {
      vec3 radiance = texelFetch(ray_history_tx, texel_hit, 0).rgb;
      history_weigh_and_accumulate(
          radiance, rgb_min, rgb_max, rgb_mean, rgb_deviation, radiance_accum, weight_accum);
    }
#endif

    radiance_accum *= safe_rcp(weight_accum);
  }

  /* Save linear depth in alpha to speed-up the bilateral filter. */
  imageStore(out_history_img, texel, vec4(radiance_accum, -vP.z));
  imageStore(out_variance_img, texel, vec4(variance, 0.0, 0.0, 0.0));
}
