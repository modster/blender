
/**
 * Diffuse Irradiance encoding, decoding, loading, evaluation functions.
 **/

#pragma BLENDER_REQUIRE(common_math_lib.glsl)

/* ---------------------------------------------------------------------- */
/** \name Visibility
 *
 * Irradiance grid is backed by a grid of small filtered distance map that reduces
 * light leak by performing a chebishev test.
 *
 * The technique is similar to the one in the paper even if not using raytracing to update
 * lightprobes.
 * "Dynamic Diffuse Global Illumination with Ray-Traced Irradiance Fields"
 * http://jcgt.org/published/0008/02/01/
 * \{ */

vec4 visibility_encode(vec2 accum, float range)
{
  accum /= range;
  vec4 data;
  data.x = fract(accum.x);
  data.y = floor(accum.x) / 255.0;
  data.z = fract(accum.y);
  data.w = floor(accum.y) / 255.0;
  return data;
}

vec2 visibility_decode(vec4 data, float range)
{
  return (data.xz + data.yw * 255.0) * range;
}

vec2 visibility_mapping_octahedron(vec3 cubevec, vec2 texel_size)
{
  /* Projection onto octahedron. */
  cubevec /= dot(vec3(1.0), abs(cubevec));
  /* Out-folding of the downward faces. */
  if (cubevec.z < 0.0) {
    vec2 cubevec_sign = step(0.0, cubevec.xy) * 2.0 - 1.0;
    cubevec.xy = (1.0 - abs(cubevec.yx)) * cubevec_sign;
  }
  /* Mapping to [0;1]ˆ2 texture space. */
  vec2 uvs = cubevec.xy * (0.5) + 0.5;
  /* Edge filtering fix. */
  uvs = (1.0 - 2.0 * texel_size) * uvs + texel_size;
  return uvs;
}

/* Returns the cell weight using the visibility data and a smooth test. */
float visibility_load_cell(IrradianceInfoData info,
                           sampler2DArray irradiance_tx,
                           int cell,
                           vec3 L,
                           float dist,
                           float bias,
                           float bleed_bias,
                           float range)
{
  /* Keep in sync with diffuse_filter_probe(). */
  ivec2 cell_co = ivec2(info.visibility_size);
  cell_co.x *= (cell % info.visibility_cells_per_row);
  cell_co.y *= (cell % info.visibility_cells_per_layer) / info.visibility_cells_per_row;
  float layer = 1.0 + float((cell / info.visibility_cells_per_layer));

  vec2 texel_size = 1.0 / vec2(textureSize(irradiance_tx, 0).xy);
  vec2 co = vec2(cell_co) * texel_size;

  vec2 uv = visibility_mapping_octahedron(-L, vec2(1.0 / float(info.visibility_size)));
  uv *= vec2(info.visibility_size) * texel_size;

  vec4 data = texture(irradiance_tx, vec3(co + uv, layer));

  /* Decoding compressed data. */
  vec2 moments = visibility_decode(data, range);
  /* Doing chebishev test. */
  float variance = abs(moments.x * moments.x - moments.y);
  variance = max(variance, bias / 10.0);

  float d = dist - moments.x;
  float p_max = variance / (variance + d * d);

  /* Increase contrast in the weight by squaring it */
  p_max *= p_max;
  /* Now reduce light-bleeding by removing the [0, x] tail and linearly rescaling (x, 1] */
  p_max = clamp((p_max - bleed_bias) / (1.0 - bleed_bias), 0.0, 1.0);

  return (dist <= moments.x) ? 1.0 : p_max;
}

/** \} */

/* ---------------------------------------------------------------------- */
/** \name Irradiance
 *
 * Using HalfLife2 ambient cube. We encode data using RGBE compression.
 * https://cdn.cloudflare.steamstatic.com/apps/valve/2006/SIGGRAPH06_Course_ShadingInValvesSourceEngine.pdf
 * \{ */

vec4 irradiance_encode(vec3 rgb)
{
  float maxRGB = max_v3(rgb);
  float fexp = ceil(log2(maxRGB));
  return vec4(rgb / exp2(fexp), (fexp + 128.0) / 255.0);
}

vec3 irradiance_decode(vec4 data)
{
  float fexp = data.a * 255.0 - 128.0;
  return data.rgb * exp2(fexp);
}

/* Samples an irradiance grid cell in the given direction. */
vec3 irradiance_load_cell(IrradianceInfoData info, sampler2DArray irradiance_tx, int cell, vec3 N)
{
  ivec2 cell_co = ivec2(3, 2);
  cell_co.x *= cell % info.irradiance_cells_per_row;
  cell_co.y *= cell / info.irradiance_cells_per_row;

  ivec3 is_negative = ivec3(step(0.0, -N));

  /* Listing 1. */
  vec3 n_squared = N * N;
  vec3 irradiance;
  vec4 data;
  data = texelFetch(irradiance_tx, ivec3(cell_co + ivec2(0, is_negative.x), 0), 0);
  irradiance = n_squared.x * irradiance_decode(data);

  data = texelFetch(irradiance_tx, ivec3(cell_co + ivec2(1, is_negative.y), 0), 0);
  irradiance += n_squared.y * irradiance_decode(data);

  data = texelFetch(irradiance_tx, ivec3(cell_co + ivec2(2, is_negative.z), 0), 0);
  irradiance += n_squared.z * irradiance_decode(data);
  return irradiance;
}

/** \} */
