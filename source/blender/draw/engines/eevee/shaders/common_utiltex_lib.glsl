
#pragma BLENDER_REQUIRE(bsdf_common_lib.glsl)

/* ---------------------------------------------------------------------- */
/** \name Utiltex
 *
 * Utiltex is a sampler2DArray that stores a number of useful small utilitary textures and lookup
 * tables.
 * \{ */

uniform sampler2DArray utilTex;

#define LUT_SIZE 64

#define LTC_MAT_LAYER 0
#define LTC_BRDF_LAYER 1
#define BRDF_LUT_LAYER 1
#define NOISE_LAYER 2
#define LTC_DISK_LAYER 3 /* UNUSED */

/* Layers 4 to 20 are for BTDF Lut. */
const float lut_btdf_layer_first = 4.0;
const float lut_btdf_layer_count = 16.0;

#define texelfetch_noise_tex(coord) \
  texelFetch(utilTex, ivec3(ivec2(coord) % LUT_SIZE, NOISE_LAYER), 0)

/* Return texture coordinates to sample Surface LUT. */
vec2 lut_coords(float cos_theta, float roughness)
{
  vec2 coords = vec2(roughness, sqrt(1.0 - cos_theta));

  /* scale and bias coordinates, for correct filtered lookup */
  return coords * (LUT_SIZE - 1.0) / LUT_SIZE + 0.5 / LUT_SIZE;
}

/* Returns the GGX split-sum precomputed in LUT. */
vec2 brdf_lut(float cos_theta, float roughness)
{
  return textureLod(utilTex, vec3(lut_coords(cos_theta, roughness), BRDF_LUT_LAYER), 0.0).rg;
}

/* Returns GGX BTDF in first component and fresnel in second. */
vec2 btdf_lut(float cos_theta, float roughness, float ior)
{
  if (ior >= 1.0) {
    vec2 split_sum = brdf_lut(cos_theta, roughness);
    float f0 = f0_from_ior(ior);
    float fresnel = F_brdf_single_scatter(vec3(f0), vec3(1.0), split_sum).r;
    /* Setting the BTDF to one is not really important since it is only used for multiscatter
     * and it's already quite close to ground truth. */
    float btdf = 1.0;
    return vec2(btdf, fresnel);
  }

  /* ior is sin of critical angle. */
  float critical_cos = sqrt(1.0 - ior * ior);

  vec3 coords;
  coords.x = sqr(ior);
  coords.y = sqrt(1.0 + critical_cos - cos_theta);
  coords.y = (coords.y - 0.05) / 1.45;
  coords.z = roughness;

  coords = saturate(coords);

  coords.xy = coords.xy * (LUT_SIZE - 1.0) / LUT_SIZE + 0.5 / LUT_SIZE;
  /* Bias the lookup in the NV direction to be able to do the clear cut
   * at the end of the function. */
  float clear_cut = saturate(roughness * lut_btdf_layer_count * 0.5);
  coords.y -= clear_cut * 0.5 / LUT_SIZE;

  float layer = coords.z * lut_btdf_layer_count;
  float layer_floored = floor(layer);

  coords.z = lut_btdf_layer_first + layer_floored;
  vec2 btdf_low = textureLod(utilTex, coords, 0.0).rg;

  coords.z += 1.0;
  vec2 btdf_high = textureLod(utilTex, coords, 0.0).rg;

  /* Manual trilinear interpolation. */
  vec2 btdf = mix(btdf_low, btdf_high, layer - layer_floored);

  /* Do a manual trim if roughness is low enough to avoid seeing the bilinear interpolation. */
  if (clear_cut < 1.0 && cos_theta < critical_cos) {
    btdf = mix(vec2(0.0, 1.0), btdf, clear_cut);
  }

  return btdf;
}

/** \} */
