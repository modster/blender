/**
 * Ray generation routines for each BSDF types.
 */

#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_bsdf_sampling_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_gbuffer_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_raytrace_trace_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)

/* Returns viewspace ray. */
Ray raytrace_create_reflection_ray(
    SamplingData sampling, vec2 noise, ClosureReflection reflection, vec3 V, vec3 P, out float pdf)
{
  vec2 noise_offset = sampling_rng_2D_get(sampling, SAMPLING_RAYTRACE_U);
  vec3 Xi = sample_cylinder(fract(noise_offset + noise));

  float roughness_sqr = max(1e-3, sqr(reflection.roughness));
  /* Gives *perfect* reflection for very small roughness. */
  if (reflection.roughness < 0.0016) {
    Xi = vec3(0.0);
  }

  vec3 T, B, N = reflection.N;
  make_orthonormal_basis(N, T, B);

  Ray ray;
  ray.origin = P;
  ray.direction = sample_ggx_reflect(Xi, roughness_sqr, V, N, T, B, pdf);
  return ray;
}

Ray raytrace_create_refraction_ray(
    SamplingData sampling, vec2 noise, ClosureRefraction refraction, vec3 V, vec3 P, out float pdf)
{
  vec2 noise_offset = sampling_rng_2D_get(sampling, SAMPLING_RAYTRACE_U);
  vec3 Xi = sample_cylinder(fract(noise_offset + noise));

  float roughness_sqr = max(1e-3, sqr(refraction.roughness));
  /* Gives *perfect* refraction for very small roughness. */
  if (refraction.roughness < 0.0016) {
    Xi = vec3(0.0);
  }
  vec3 T, B, N = refraction.N;
  make_orthonormal_basis(N, T, B);

  Ray ray;
  ray.origin = P;
  ray.direction = sample_ggx_refract(Xi, roughness_sqr, refraction.ior, V, N, T, B, pdf);
  return ray;
}

Ray raytrace_create_diffuse_ray(
    SamplingData sampling, vec2 noise, ClosureDiffuse diffuse, vec3 P, out float pdf)
{
  vec2 noise_offset = sampling_rng_2D_get(sampling, SAMPLING_RAYTRACE_U);
  vec3 Xi = sample_cylinder(fract(noise_offset + noise));

  /* Bias the rays so we never get really high energy rays almost parallel to the surface. */
  Xi.x = Xi.x * 0.98 + 0.02;

  vec3 T, B, N = diffuse.N;
  make_orthonormal_basis(N, T, B);

  Ray ray;
  ray.origin = P;
  ray.direction = sample_cosine_hemisphere(Xi, N, T, B, pdf);
  return ray;
}
