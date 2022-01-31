
#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_geom_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_closure_lib.glsl)

/* -------------------------------------------------------------------- */
/** \name Encoding and decoding functions
 *
 * \{ */

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
vec2 gbuffer_encode_normal(vec3 normal)
{
  vec3 vN = normal_world_to_view(normal);
  bool neg = vN.z < 0.0;
  if (neg) {
    vN.z = -vN.z;
  }
  vec2 packed_normal = normal_encode(vN);
  // return packed_normal;
  return (neg) ? -packed_normal : packed_normal;
}

vec3 gbuffer_decode_normal(vec2 packed_normal)
{
  bool neg = packed_normal.y < 0.0;
  vec3 vN = normal_decode(abs(packed_normal));
  if (neg) {
    vN.z = -vN.z;
  }
  return normal_view_to_world(vN);
}

/* Note: does not handle negative colors. */
uint gbuffer_encode_color(vec3 color)
{
  color *= 1.0; /* Test */
  float intensity = length(color);
  /* Normalize to store it like a normal vector. */
  // color *= safe_rcp(intensity);

  uint encoded_color;
  // encoded_color = gbuffer_encode_unit_float_to_uint(saturate(color.x), 10u) << 10u;
  // encoded_color |= gbuffer_encode_unit_float_to_uint(saturate(color.y), 10u);
  // encoded_color |= gbuffer_encode_unit_float_to_uint(saturate(intensity), 12u) << 20u;

  encoded_color = gbuffer_encode_unit_float_to_uint(saturate(color.x), 11u);
  encoded_color |= gbuffer_encode_unit_float_to_uint(saturate(color.y), 11u) << 11u;
  encoded_color |= gbuffer_encode_unit_float_to_uint(saturate(color.z), 10u) << 21u;
  return encoded_color;
}

vec3 gbuffer_decode_color(uint packed_data)
{
  vec3 color;
  // color.x = gbuffer_decode_unit_float_from_uint(packed_data >> 10u, 10u);
  // color.y = gbuffer_decode_unit_float_from_uint(packed_data, 10u);
  // color.z = sqrt(1.0 - clamp(dot(color.xy, color.xy), 0.0, 1.0));
  // color *= gbuffer_decode_unit_float_from_uint(packed_data >> 20u, 12u);

  color.x = gbuffer_decode_unit_float_from_uint(packed_data, 11u);
  color.y = gbuffer_decode_unit_float_from_uint(packed_data >> 11u, 11u);
  color.z = gbuffer_decode_unit_float_from_uint(packed_data >> 21u, 10u);
  color *= 1.0; /* Test */
  return color;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Global data
 *
 * \{ */

void gbuffer_load_global_data(vec4 transmit_normal_in, out float thickness)
{
  thickness = transmit_normal_in.w;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Diffuse data
 *
 * \{ */

ClosureDiffuse gbuffer_load_diffuse_data(vec4 color_in, vec4 normal_in, vec4 data_in)
{
  ClosureDiffuse data_out;
  if (normal_in.z == -1.0) {
    /* Transmission data is Refraction data. */
    data_out.color = vec3(0.0);
    data_out.N = vec3(1.0);
    data_out.sss_id = 0u;
    data_out.sss_radius = vec3(-1.0);
  }
  else {
    data_out.color = color_in.rgb;
    data_out.N = gbuffer_decode_normal(normal_in.xy);
    data_out.sss_id = uint(normal_in.z * 1024.0);
    data_out.sss_radius = data_in.rgb;
  }
  return data_out;
}

ClosureDiffuse gbuffer_load_diffuse_data(sampler2D transmit_color_tx,
                                         sampler2D transmit_normal_tx,
                                         sampler2D transmit_data_tx,
                                         vec2 uv)
{
  vec4 tra_col_in = texture(transmit_color_tx, uv);
  vec4 tra_nor_in = texture(transmit_normal_tx, uv);
  vec4 tra_dat_in = texture(transmit_data_tx, uv);
  return gbuffer_load_diffuse_data(tra_col_in, tra_nor_in, tra_dat_in);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Glossy data
 *
 * \{ */

ClosureReflection gbuffer_load_reflection_data(vec4 color_in, vec4 normal_in)
{
  ClosureReflection data_out;
  data_out.color = color_in.rgb;
  data_out.N = gbuffer_decode_normal(normal_in.xy);
  data_out.roughness = normal_in.z;
  return data_out;
}

ClosureReflection gbuffer_load_reflection_data(sampler2D reflect_color_tx,
                                               sampler2D reflect_normal_tx,
                                               vec2 uv)
{
  vec4 ref_col_in = texture(reflect_color_tx, uv);
  vec4 ref_nor_in = texture(reflect_normal_tx, uv);
  return gbuffer_load_reflection_data(ref_col_in, ref_nor_in);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Refraction data
 *
 * \{ */

ClosureRefraction gbuffer_load_refraction_data(vec4 color_in, vec4 normal_in, vec4 data_in)
{
  ClosureRefraction data_out;
  if (normal_in.z == -1.0) {
    data_out.color = color_in.rgb;
    data_out.N = gbuffer_decode_normal(normal_in.xy);
    data_out.ior = data_in.x;
    data_out.roughness = data_in.y;
  }
  else {
    /* Transmission data is Diffuse/SSS data. */
    data_out.color = vec3(0.0);
    data_out.N = vec3(1.0);
    data_out.ior = -1.0;
    data_out.roughness = 0.0;
  }
  return data_out;
}

ClosureRefraction gbuffer_load_refraction_data(sampler2D transmit_color_tx,
                                               sampler2D transmit_normal_tx,
                                               sampler2D transmit_data_tx,
                                               vec2 uv)
{
  vec4 tra_col_in = texture(transmit_color_tx, uv);
  vec4 tra_nor_in = texture(transmit_normal_tx, uv);
  vec4 tra_dat_in = texture(transmit_data_tx, uv);
  return gbuffer_load_refraction_data(tra_col_in, tra_nor_in, tra_dat_in);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Volume data
 *
 * Pack all volumetric effects.
 *
 * \{ */

#define VOLUME_HETEROGENEOUS -2.0

uvec4 gbuffer_store_volume_data(ClosureVolume data_in)
{
  uvec4 data_out;
  data_out.x = gbuffer_encode_color(data_in.emission);
  data_out.y = gbuffer_encode_color(data_in.scattering);
  data_out.z = gbuffer_encode_color(data_in.transmittance);
  data_out.w = floatBitsToUint(data_in.anisotropy);
  return data_out;
}

ClosureVolume gbuffer_load_volume_data(usampler2D gbuffer_tx, vec2 uv)
{
  uvec4 data_in = texture(gbuffer_tx, uv);

  ClosureVolume data_out;
  data_out.emission = gbuffer_decode_color(data_in.x);
  data_out.scattering = gbuffer_decode_color(data_in.y);
  data_out.transmittance = gbuffer_decode_color(data_in.z);
  data_out.anisotropy = uintBitsToFloat(data_in.w);
  return data_out;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Emission data
 *
 * \{ */

vec3 gbuffer_store_emission_data(ClosureEmission data_in)
{
  return data_in.emission;
}

ClosureEmission gbuffer_load_emission_data(sampler2D gbuffer_tx, vec2 uv)
{
  vec4 data_in = texture(gbuffer_tx, uv);

  ClosureEmission data_out;
  data_out.emission = data_in.xyz;
  return data_out;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Transparency data
 *
 * \{ */

vec4 gbuffer_store_transparency_data(ClosureTransparency data_in)
{
  vec4 data_out;
  data_out.xyz = data_in.transmittance;
  data_out.w = data_in.holdout;
  return data_out;
}

ClosureTransparency gbuffer_load_transparency_data(sampler2D gbuffer_tx, vec2 uv)
{
  vec4 data_in = texture(gbuffer_tx, uv);

  ClosureTransparency data_out;
  data_out.transmittance = data_in.xyz;
  data_out.holdout = data_in.w;
  return data_out;
}

/** \} */
