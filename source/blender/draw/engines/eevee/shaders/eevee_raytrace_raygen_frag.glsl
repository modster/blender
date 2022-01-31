/**
 * Simple wrapper around the screen-space raytracing routine.
 * The goal is to output the tracing result buffer that can be denoised.
 * We fallback to reflection probe if the ray fails.
 */

#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_gbuffer_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_raytrace_raygen_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_raytrace_trace_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_sampling_lib.glsl)

layout(std140) uniform sampling_block
{
  SamplingData sampling;
};

layout(std140) uniform raytrace_block
{
  RaytraceData raytrace;
};

layout(std140) uniform hiz_block
{
  HiZData hiz;
};

layout(std140) uniform cubes_block
{
  CubemapData cubes[CULLING_ITEM_BATCH];
};

layout(std140) uniform lightprobes_info_block
{
  LightProbeInfoData probes_info;
};

uniform sampler2D hiz_tx;
uniform sampler2D hiz_front_tx;
uniform samplerCubeArray lightprobe_cube_tx;
uniform sampler2D radiance_tx;
uniform sampler2D combined_tx;
uniform sampler2D cl_color_tx;
uniform sampler2D cl_normal_tx;
uniform sampler2D cl_data_tx;
uniform sampler2DArray utility_tx;

utility_tx_fetch_define(utility_tx);
utility_tx_sample_define(utility_tx);

in vec4 uvcoordsvar;

layout(location = 0) out vec4 out_ray_data;
layout(location = 1) out vec4 out_ray_radiance;

/* Prototypes. */
vec3 lightprobe_cubemap_eval(vec3 P, vec3 R, float roughness, float random_threshold);

void main()
{
  vec2 uv = uvcoordsvar.xy;
  float gbuffer_depth = texelFetch(hiz_front_tx, ivec2(gl_FragCoord.xy), 0).r;
  vec3 P = get_world_space_from_depth(uv, gbuffer_depth);
  vec3 V = cameraVec(P);

  vec4 noise = utility_tx_fetch(gl_FragCoord.xy, UTIL_BLUE_NOISE_LAYER).gbar;

  out_ray_data = vec4(0.0);
  out_ray_radiance = vec4(0.0);

#if defined(DIFFUSE)
  vec4 tra_col_in = vec4(0.0); /* UNUSED */
  vec4 tra_nor_in = texture(cl_normal_tx, uv);
  vec4 tra_dat_in = vec4(0.0); /* UNUSED */

  ClosureDiffuse diffuse = gbuffer_load_diffuse_data(tra_col_in, tra_nor_in, tra_dat_in);

  if (diffuse.sss_radius.r < 0.0) {
    /* Refraction pixel. */
    return;
  }
  float roughness = 1.0;

#elif defined(REFRACTION)
  vec4 tra_col_in = vec4(0.0); /* UNUSED */
  vec4 tra_nor_in = texture(cl_normal_tx, uv);
  vec4 tra_dat_in = texture(cl_data_tx, uv);

  ClosureRefraction refraction = gbuffer_load_refraction_data(tra_col_in, tra_nor_in, tra_dat_in);

  float thickness;
  gbuffer_load_global_data(tra_nor_in, thickness);

  if (refraction.ior == -1.0) {
    /* Diffuse/SSS pixel. */
    return;
  }
  float roughness = refraction.roughness;

#else
  ClosureReflection reflection = gbuffer_load_reflection_data(cl_color_tx, cl_normal_tx, uv);

  float roughness = reflection.roughness;
#endif

  /* Generate ray. */
  float pdf;
#if defined(DIFFUSE)
  Ray ray = raytrace_create_diffuse_ray(sampling, noise.xy, diffuse, P, pdf);
#elif defined(REFRACTION)
  Ray ray = raytrace_create_refraction_ray(sampling, noise.xy, refraction, V, P, pdf);
#else
  Ray ray = raytrace_create_reflection_ray(sampling, noise.xy, reflection, V, P, pdf);
#endif
  ray = raytrace_world_ray_to_view(ray);

  bool hit = false;

#ifndef SKIP_TRACE
  vec2 noise_offset = sampling_rng_2D_get(sampling, SAMPLING_RAYTRACE_W);
  vec2 rand = fract(noise.zw + noise_offset.xy);
  /* Extend the ray to cover the whole view. */
  ray.direction *= 1e16;

  if (roughness - rand.y * 0.2 < raytrace.max_roughness) {
#  ifdef REFRACTION
    const bool discard_backface = false;
    const bool allow_self_intersection = true;
    /* TODO(fclem): Take IOR into account in the roughness LOD bias. */
#  else
    const bool discard_backface = true;
    const bool allow_self_intersection = false;
#  endif
    hit = raytrace_screen(
        raytrace, hiz, hiz_tx, rand.x, roughness, discard_backface, allow_self_intersection, ray);
  }
#endif

  vec3 radiance;
  if (hit) {
    /* Evaluate radiance at hitpoint. */
    vec2 hit_uv = get_uvs_from_view(ray.origin + ray.direction);

    radiance = textureLod(radiance_tx, hit_uv, 0.0).rgb;

#if defined(DIFFUSE)
    /* For diffuse, the radiance is still split and SSS materials don't have the color applied. */
    vec4 tra_col_in = texture(cl_color_tx, hit_uv);
    vec4 tra_nor_in = texture(cl_normal_tx, hit_uv);
    vec4 tra_dat_in = vec4(0.0); /* UNUSED */
    ClosureDiffuse hit_diffuse = gbuffer_load_diffuse_data(tra_col_in, tra_nor_in, tra_dat_in);

    if (hit_diffuse.sss_id > 0u) {
      /* Refraction pixel. */
      radiance *= hit_diffuse.color;
    }
    radiance += textureLod(combined_tx, hit_uv, 0.0).rgb;
#endif
  }
  else {
    /* Evaluate fallback lightprobe. */
    float noise_offset = sampling_rng_1D_get(sampling, SAMPLING_LIGHTPROBE);
    float random_probe = fract(noise.w + noise_offset);

    ray = raytrace_view_ray_to_world(ray);
    /* TOOD(fclem): We could reduce noise by mapping ray pdf to roughness. */
    float roughness = 0.0;

    radiance = lightprobe_cubemap_eval(ray.origin, ray.direction, roughness, random_probe);
  }
  /* Apply brightness clamping. */
  float luma = max_v3(radiance);
  radiance *= 1.0 - max(0.0, luma - raytrace.brightness_clamp) * safe_rcp(luma);
  /* Limit to the smallest non-0 value that the format can encode.
   * Strangely it does not correspond to the IEEE spec. */
  float inv_pdf = (pdf == 0.0) ? 0.0 : max(6e-8, 1.0 / pdf);
  float hit_depth = saturate(get_depth_from_view_z(ray.origin.z + ray.direction.z));
  /* Output the ray. */
  out_ray_data = vec4(ray.direction, inv_pdf);
  out_ray_radiance = vec4(radiance, hit_depth);
}

#pragma BLENDER_REQUIRE_POST(eevee_lightprobe_eval_cubemap_lib.glsl)
