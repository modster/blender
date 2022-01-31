
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

#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_bsdf_microfacet_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_gbuffer_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_sampling_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)

layout(std140) uniform hiz_block
{
  HiZData hiz;
};

uniform sampler2D ray_radiance_tx;
uniform sampler2D ray_variance_tx;
uniform sampler2D cl_color_tx;
uniform sampler2D cl_normal_tx;
uniform sampler2D cl_data_tx;

in vec4 uvcoordsvar;

layout(location = 0) out vec4 out_combined;
layout(location = 1) out vec4 out_diffuse;
layout(location = 2) out vec3 out_specular;

#if defined(DIFFUSE)
#  define RADIUS 4
#elif defined(REFRACTION)
#  define RADIUS 1
#else
#  define RADIUS 1
#endif

float normal_pdf(float x_sqr, float sigma_inv, float sigma_inv_sqr)
{
  return exp(-0.5 * x_sqr * sigma_inv_sqr) * sigma_inv;
}

void main(void)
{
  vec2 uv = uvcoordsvar.xy;
  float ray_variance = texture(ray_variance_tx, uv).r;
  vec4 ray_data = texture(ray_radiance_tx, uv);
  float center_depth = ray_data.w;
  vec2 texel_size = hiz.pixel_to_ndc * 0.5;

  out_combined = vec4(0.0);
  out_diffuse = vec4(0.0);
  out_specular = vec3(0.0);

#if defined(DIFFUSE)
  ClosureDiffuse closure = gbuffer_load_diffuse_data(cl_color_tx, cl_normal_tx, cl_data_tx, uv);
  if (closure.sss_radius.r < 0.0) {
    return;
  }
  float sigma_pixel = 3.0;
#elif defined(REFRACTION)
  ClosureRefraction closure = gbuffer_load_refraction_data(
      cl_color_tx, cl_normal_tx, cl_data_tx, uv);
  if (closure.ior == -1.0) {
    return;
  }
  float sigma_pixel = 1.0;
#else
  ClosureReflection closure = gbuffer_load_reflection_data(cl_color_tx, cl_normal_tx, uv);
  float sigma_pixel = 1.0;
#endif
  /* TODO(fclem): Sigma based on variance. */
  float sigma_depth = 0.1; /* TODO user option? */

  float px_sigma_inv = 1.0 / sigma_pixel;
  float px_sigma_inv_sqr = sqr(px_sigma_inv);
  float depth_sigma_inv = 1.0 / sigma_depth;
  float depth_sigma_inv_sqr = sqr(depth_sigma_inv);

  float weight_accum = normal_pdf(0.0, px_sigma_inv, px_sigma_inv_sqr);
  vec3 radiance_accum = ray_data.rgb * weight_accum;
  for (int x = -RADIUS; x <= RADIUS; x++) {
    for (int y = -RADIUS; y <= RADIUS; y++) {
      /* Skip center pixels. */
      if (x == 0 && y == 0) {
        continue;
      }
      vec2 sample_uv = uv + vec2(x, y) * texel_size;
      vec4 ray_data = texture(ray_radiance_tx, sample_uv);
      /* Skip unprocessed pixels. */
      if (ray_data.w == 0.0) {
        continue;
      }
      float delta_pixel_sqr = len_squared(vec2(x, y));
      float delta_depth_sqr = sqr(abs(center_depth - ray_data.w));
      /* TODO(fclem): OPTI might be a good idea to compare view normal to avoid one matrix mult. */
      vec3 sample_N = gbuffer_decode_normal(texture(cl_normal_tx, sample_uv).xy);
      /* Bilateral weight. */
      float weight = saturate(dot(sample_N, closure.N)) *
                     normal_pdf(delta_pixel_sqr, px_sigma_inv, px_sigma_inv_sqr) *
                     normal_pdf(delta_depth_sqr, depth_sigma_inv, depth_sigma_inv_sqr);

      radiance_accum += ray_data.rgb * weight;
      weight_accum += weight;
    }
  }
  radiance_accum *= safe_rcp(weight_accum);
  radiance_accum *= closure.color;

  out_combined = vec4(radiance_accum, 0.0);
#if defined(DIFFUSE)
  out_diffuse.rgb = radiance_accum;
#else
  out_specular = radiance_accum;
#endif
}
