
/* Lib file to include to evaluate a nodetree without evaluating BSDFs. */

/* -------------------------------------------------------------------- */
/** \name Utility functions to work with BSDFs
 * \{ */

vec3 diffuse_dominant_dir(vec3 bent_normal)
{
  return vec3(0);
}
vec3 specular_dominant_dir(vec3 N, vec3 V, float roughness)
{
  return vec3(0.0);
}
vec3 refraction_dominant_dir(vec3 N, vec3 V, float roughness, float ior)
{
  return vec3(0.0);
}
float F_eta(float eta, float cos_theta)
{
  return 0.0;
}
vec3 F_brdf_single_scatter(vec3 f0, vec3 f90, vec2 lut)
{
  return vec3(0);
}
vec3 F_brdf_multi_scatter(vec3 f0, vec3 f90, vec2 lut)
{
  return vec3(0);
}
vec2 brdf_lut(float cos_theta, float roughness)
{
  return vec2(0);
}
vec2 btdf_lut(float cos_theta, float roughness, float ior)
{
  return vec2(0);
}

/** \} */
