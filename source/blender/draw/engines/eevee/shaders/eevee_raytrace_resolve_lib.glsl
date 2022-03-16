
/**
 * Final denoise step using a bilateral filter. Filter radius is controled by variance estimate.
 *
 * Inputs: Ray radiance (denoised), Mean hit depth, Extimated variance from ray reconstruction
 * Outputs: Ray radiance (filtered).
 *
 * Linear depth is packed in ray_radiance for this step.
 * Following "Stochastic All The Things: Raytracing in Hybrid Real-Time Rendering"
 * by Tomasz Stachowiak
 * https://www.ea.com/seed/news/seed-dd18-presentation-slides-raytracing
 */

/* Requires raytrace_buffer_buf in resources. */

#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_gbuffer_lib.glsl)

float normal_pdf(float x_sqr, float sigma_inv, float sigma_inv_sqr)
{
  return exp(-0.5 * x_sqr * sigma_inv_sqr) * sigma_inv;
}

/* Resolve a raytrace buffer using a bilateral filter with depth + 2D + normal threshold.
 * \a texel_origin: fullres input texel.
 * \a kernel_radius: the radius of the filer.
 * \a sigma_pixel: filter sigma to apply in pixel space.
 * \return : Resolved indirect radiance.
 */
vec3 raytrace_resolve(ivec2 texel_origin,
                      const int kernel_radius,
                      const float sigma_pixel,
                      sampler2D closure_normal_tx,
                      sampler2D ray_data_tx,
                      sampler2D ray_radiance_tx)
{
  /* Convert to half res pixel. */
  ivec2 texel = texel_origin / raytrace_buffer_buf.res_scale;
  ivec2 texel_fullres = texel * raytrace_buffer_buf.res_scale + raytrace_buffer_buf.res_bias;

  vec4 ray_data = texelFetch(ray_radiance_tx, texel, 0);
  float center_depth = ray_data.w;

  vec2 closure_N_packed = texelFetch(closure_normal_tx, texel_fullres, 0).xy;
  vec3 closure_N = gbuffer_decode_normal_view(closure_N_packed);

  /* TODO(fclem): Sigma based on variance. */
  float sigma_depth = 0.1; /* TODO user option? */

  float px_sigma_inv = 1.0 / sigma_pixel;
  float px_sigma_inv_sqr = sqr(px_sigma_inv);
  float depth_sigma_inv = 1.0 / sigma_depth;
  float depth_sigma_inv_sqr = sqr(depth_sigma_inv);

  float weight_accum = normal_pdf(0.0, px_sigma_inv, px_sigma_inv_sqr) *
                       normal_pdf(0.0, depth_sigma_inv, depth_sigma_inv_sqr);
  vec3 radiance_accum = ray_data.rgb * weight_accum;
  for (int x = -kernel_radius; x <= kernel_radius; x++) {
    for (int y = -kernel_radius; y <= kernel_radius; y++) {
      /* Skip center pixels. */
      if (x == 0 && y == 0) {
        continue;
      }
      ivec2 texel_sample = texel + ivec2(x, y);
      vec4 ray_data_sample = texelFetch(ray_radiance_tx, texel_sample, 0);

      float delta_pixel_sqr = len_squared(vec2(x, y));
      float delta_depth_sqr = sqr(abs(center_depth - ray_data_sample.w));
      /* Bilateral weight. */
      float weight = normal_pdf(delta_pixel_sqr, px_sigma_inv, px_sigma_inv_sqr) *
                     normal_pdf(delta_depth_sqr, depth_sigma_inv, depth_sigma_inv_sqr);
      /* Avoid bluring accross surfaces with different normals. */
      ivec2 texel_fullres_sample = texel_sample * raytrace_buffer_buf.res_scale +
                                   raytrace_buffer_buf.res_bias;
      vec2 sample_N_packed = texelFetch(closure_normal_tx, texel_fullres_sample, 0).xy;
      vec3 sample_N = gbuffer_decode_normal_view(sample_N_packed);
      weight *= saturate(dot(sample_N, closure_N));

      radiance_accum += ray_data_sample.rgb * weight;
      weight_accum += weight;
    }
  }
  return radiance_accum * safe_rcp(weight_accum);
}
