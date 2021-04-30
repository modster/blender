
#pragma BLENDER_REQUIRE(common_view_lib.glsl)

uint gbuffer_encode_unit_float_to_uint(float scalar, const uint bit_size)
{
  float fac = float((1u << bit_size) - 1u);
  return uint(saturate(scalar) * fac);
}

float gbuffer_decode_unit_float_from_uint(uint packed_scalar, const uint bit_size)
{
  float fac = 1.0 / float((1u << bit_size) - 1u);
  uint mask = ~(0xFFFFFFFFu << bit_size);
  return float(packed_scalar & mask) * fac;
}

/* Expects input to be normalized. */
uint gbuffer_encode_normal(vec3 normal)
{
  normal = normal_world_to_view(normal);
  /* TODO spheremap transformation */
  normal = normal * 0.5 + 0.5;

  uint encoded_normal;
  encoded_normal = gbuffer_encode_unit_float_to_uint(normal.x, 10u) << 10u;
  encoded_normal |= gbuffer_encode_unit_float_to_uint(normal.y, 10u);
  return encoded_normal;
}

vec3 gbuffer_decode_normal(uint packed_normal)
{
  vec3 decoded_normal;
  decoded_normal.x = gbuffer_decode_unit_float_from_uint(packed_normal >> 10u, 10u);
  decoded_normal.y = gbuffer_decode_unit_float_from_uint(packed_normal, 10u);

  decoded_normal.xy = decoded_normal.xy * 2.0 - 1.0;
  decoded_normal.z = sqrt(1.0 - clamp(dot(decoded_normal.xy, decoded_normal.xy), 0.0, 1.0));

  /* TODO spheremap transformation */

  decoded_normal = normal_view_to_world(decoded_normal);
  return decoded_normal;
}

/* Note: does not handle negative colors. */
uint gbuffer_encode_color(vec3 color)
{
  float intensity = length(color);
  /* Normalize to store it like a normal vector. */
  color *= safe_rcp(intensity);

  uint encoded_color;
  encoded_color = gbuffer_encode_unit_float_to_uint(saturate(color.x), 10u) << 10u;
  encoded_color |= gbuffer_encode_unit_float_to_uint(saturate(color.y), 10u);
  encoded_color |= gbuffer_encode_unit_float_to_uint(saturate(intensity), 12u) << 20u;
  return encoded_color;
}

vec3 gbuffer_decode_color(uint packed_data)
{
  vec3 color;
  color.x = gbuffer_decode_unit_float_from_uint(packed_data >> 10u, 10u);
  color.y = gbuffer_decode_unit_float_from_uint(packed_data, 10u);
  color.z = sqrt(1.0 - clamp(dot(color.xy, color.xy), 0.0, 1.0));
  color *= gbuffer_decode_unit_float_from_uint(packed_data >> 20u, 12u);
  return color;
}

/* -------------------------------------------------------------------- */
/** \name Diffuse data
 *
 * Pack either a Diffuse BSDF, a Subsurface BSSSDF, or a Translucent BSDF.
 *
 * \{ */

struct GBufferDiffuseData {
  vec3 color;
  vec3 N;
  float thickness;
  vec3 sss_radius;
  uint sss_id;
};

uvec4 gbuffer_encode_diffuse_data(GBufferDiffuseData data_in)
{
  uvec4 data_out;
  data_out.x = gbuffer_encode_color(data_in.color);
  data_out.y = gbuffer_encode_normal(data_in.N);
  /* TODO(fclem) High dynamic range. */
  data_out.y |= gbuffer_encode_unit_float_to_uint(saturate(data_in.thickness), 12u) << 20u;
  data_out.z = gbuffer_encode_color(data_in.sss_radius);
  data_out.w = data_in.sss_id;
  return data_out;
}

GBufferDiffuseData gbuffer_decode_diffuse_data(uvec4 data_in)
{
  GBufferDiffuseData data_out;
  data_out.color = gbuffer_decode_color(data_in.x);
  data_out.N = gbuffer_decode_normal(data_in.y);
  data_out.thickness = gbuffer_decode_unit_float_from_uint(data_in.y >> 20u, 12u);
  data_out.sss_radius = gbuffer_decode_color(data_in.z);
  data_out.sss_id = data_in.w;
  return data_out;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Glossy data
 *
 * Pack a Glossy BSDF.
 *
 * \{ */

struct GBufferReflectionData {
  vec3 color;
  vec3 N;
  float roughness;
};

uvec2 gbuffer_encode_reflection_data(GBufferReflectionData data_in)
{
  uvec2 data_out;
  data_out.x = gbuffer_encode_color(data_in.color);
  data_out.y = gbuffer_encode_normal(data_in.N);
  data_out.y |= gbuffer_encode_unit_float_to_uint(saturate(data_in.roughness), 12u) << 20u;
  return data_out;
}

GBufferReflectionData gbuffer_decode_reflection_data(uvec2 data_in)
{
  GBufferReflectionData data_out;
  data_out.color = gbuffer_decode_color(data_in.x);
  data_out.N = gbuffer_decode_normal(data_in.y);
  data_out.roughness = gbuffer_decode_unit_float_from_uint(data_in.y >> 20u, 12u);
  return data_out;
}

/** \} */
