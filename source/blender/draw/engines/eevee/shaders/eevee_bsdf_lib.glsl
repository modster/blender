
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)

/* -------------------------------------------------------------------- */
/** \name Utility functions to work with BSDFs
 * \{ */

vec3 diffuse_dominant_dir(vec3 bent_normal)
{
  return bent_normal;
}

vec3 specular_dominant_dir(vec3 N, vec3 V, float roughness)
{
  vec3 R = -reflect(V, N);
  float smoothness = 1.0 - roughness;
  float fac = smoothness * (sqrt(smoothness) + roughness);
  return normalize(mix(N, R, fac));
}

vec3 refraction_dominant_dir(vec3 N, vec3 V, float roughness, float ior)
{
  /* TODO: This a bad approximation. Better approximation should fit
   * the refracted vector and roughness into the best prefiltered reflection
   * lobe. */
  /* Correct the IOR for ior < 1.0 to not see the abrupt delimitation or the TIR */
  ior = (ior < 1.0) ? mix(ior, 1.0, roughness) : ior;
  float eta = 1.0 / ior;

  float NV = dot(N, -V);

  /* Custom Refraction. */
  float k = 1.0 - eta * eta * (1.0 - NV * NV);
  k = max(0.0, k); /* Only this changes. */
  vec3 R = eta * -V - (eta * NV + sqrt(k)) * N;

  return R;
}

float ior_from_f0(float f0)
{
  float f = sqrt(f0);
  return (-f - 1.0) / (f - 1.0);
}

/* Simplified form of F_eta(eta, 1.0). */
float f0_from_ior(float eta)
{
  float A = (eta - 1.0) / (eta + 1.0);
  return A * A;
}

/* Fresnel monochromatic, perfect mirror. */
float F_eta(float eta, float cos_theta)
{
  /* Compute fresnel reflectance without explicitly computing the refracted direction. */
  float c = abs(cos_theta);
  float g = eta * eta - 1.0 + c * c;
  if (g > 0.0) {
    g = sqrt(g);
    float A = (g - c) / (g + c);
    float B = (c * (g + c) - 1.0) / (c * (g - c) + 1.0);
    return 0.5 * A * A * (1.0 + B * B);
  }
  /* Total internal reflections. */
  return 1.0;
}

/* Fresnel color blend base on fresnel factor. */
vec3 F_color_blend(float eta, float fresnel, vec3 f0_color)
{
  float f0 = f0_from_ior(eta);
  float fac = saturate((fresnel - f0) / (1.0 - f0));
  return mix(f0_color, vec3(1.0), fac);
}

/* Fresnel split-sum approximation. */
vec3 F_brdf_single_scatter(vec3 f0, vec3 f90, vec2 lut)
{
  /* Unreal specular matching : if specular color is below 2% intensity, treat as shadowning */
  return lut.y * f90 + lut.x * f0;
}

/* Multi-scattering brdf approximation from :
 * "A Multiple-Scattering Microfacet Model for Real-Time Image-based Lighting"
 * by Carmelo J. Fdez-Agüera. */
vec3 F_brdf_multi_scatter(vec3 f0, vec3 f90, vec2 lut)
{
  vec3 FssEss = lut.y * f90 + lut.x * f0;

  float Ess = lut.x + lut.y;
  float Ems = 1.0 - Ess;
  vec3 Favg = f0 + (1.0 - f0) / 21.0;
  vec3 Fms = FssEss * Favg / (1.0 - (1.0 - Ess) * Favg);
  /* We don't do anything special for diffuse surfaces because the principle bsdf
   * does not care about energy conservation of the specular layer for dielectrics. */
  return FssEss + Fms * Ems;
}

/* GGX */
float D_ggx_opti(float NH, float a2)
{
  float tmp = (NH * a2 - NH) * NH + 1.0;
  return M_PI * tmp * tmp; /* Doing RCP and mul a2 at the end. */
}

float G1_Smith_GGX_opti(float NX, float a2)
{
  /* Using Brian Karis approach and refactoring by NX/NX
   * this way the (2*NL)*(2*NV) in G = G1(V) * G1(L) gets canceled by the brdf denominator 4*NL*NV
   * Rcp is done on the whole G later.
   * Note that this is not convenient for the transmission formula. */
  return NX + sqrt(NX * (NX - NX * a2) + a2);
  /* return 2 / (1 + sqrt(1 + a2 * (1 - NX*NX) / (NX*NX) ) ); /* Reference function. */
}

float bsdf_ggx(vec3 N, vec3 L, vec3 V, float roughness)
{
  float a = roughness;
  float a2 = a * a;

  vec3 H = normalize(L + V);
  float NH = max(dot(N, H), 1e-8);
  float NL = max(dot(N, L), 1e-8);
  float NV = max(dot(N, V), 1e-8);

  float G = G1_Smith_GGX_opti(NV, a2) * G1_Smith_GGX_opti(NL, a2); /* Doing RCP at the end */
  float D = D_ggx_opti(NH, a2);

  /* Denominator is canceled by G1_Smith */
  /* bsdf = D * G / (4.0 * NL * NV); /* Reference function. */
  return NL * a2 / (D * G); /* NL to Fit cycles Equation : line. 345 in bsdf_microfacet.h */
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Utility functions to work with BSDFs
 * \{ */

/* Same thing as Cycles without the comments to make it shorter. */
vec3 ensure_valid_reflection(vec3 Ng, vec3 I, vec3 N)
{
  vec3 R;
  float NI = dot(N, I);
  float NgR, threshold;
  /* Check if the incident ray is coming from behind normal N. */
  if (NI > 0.0) {
    /* Normal reflection. */
    R = (2.0 * NI) * N - I;
    NgR = dot(Ng, R);
    /* Reflection rays may always be at least as shallow as the incoming ray. */
    threshold = min(0.9 * dot(Ng, I), 0.01);
    if (NgR >= threshold) {
      return N;
    }
  }
  else {
    /* Bad incident. */
    R = -I;
    NgR = dot(Ng, R);
    threshold = 0.01;
  }
  /* Lift the reflection above the threshold. */
  R = R + Ng * (threshold - NgR);
  /* Find a bisector. */
  return safe_normalize(I * length(R) + R * length(I));
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Cone angle approximation
 *
 * Returns a fitted cone angle given the input roughness
 * \{ */

float cone_cosine(float r)
{
  /* Using phong gloss
   * roughness = sqrt(2/(gloss+2)) */
  float gloss = -2 + 2 / (r * r);
  /* Drobot 2014 in GPUPro5 */
  // return cos(2.0 * sqrt(2.0 / (gloss + 2)));
  /* Uludag 2014 in GPUPro5 */
  // return pow(0.244, 1 / (gloss + 1));
  /* Jimenez 2016 in Practical Realtime Strategies for Accurate Indirect Occlusion*/
  return exp2(-3.32193 * r * r);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name BRDF & BTDF Look up table functions
 *
 * BRDFs are costly to evaluated at runtime. We precompute them by using the split-sum
 * approximation method descibed by Brian Karis in "Real Shading in Unreal Engine 4" from
 * Siggraph 2013.
 * \{ */

/* Return texture coordinates to sample Surface LUT. */
vec2 lut_coords(float cos_theta, float roughness)
{
  vec2 coords = vec2(roughness, sqrt(1.0 - cos_theta));
  /* scale and bias coordinates, for correct filtered lookup */
  return coords * UTIL_TEX_UV_SCALE + UTIL_TEX_UV_BIAS;
}

/* Returns the GGX split-sum precomputed in LUT. */
vec2 brdf_lut(float cos_theta, float roughness)
{
  return utility_tx_sample(lut_coords(cos_theta, roughness), UTIL_BSDF_LAYER).rg;
}

/* Return texture coordinates to sample Surface LUT. */
vec3 lut_coords_btdf(float cos_theta, float roughness, float ior)
{
  /* ior is sin of critical angle. */
  float critical_cos = sqrt(1.0 - ior * ior);

  vec3 coords;
  coords.x = sqr(ior);
  coords.y = cos_theta;
  coords.y -= critical_cos;
  coords.y /= (coords.y > 0.0) ? (1.0 - critical_cos) : critical_cos;
  coords.y = coords.y * 0.5 + 0.5;
  coords.z = roughness;

  coords = saturate(coords);

  /* Scale and bias coordinates, for correct filtered lookup. */
  coords.xy = coords.xy * UTIL_TEX_UV_SCALE + UTIL_TEX_UV_BIAS;

  return coords;
}

/* Returns GGX BTDF in first component and fresnel in second. */
vec2 btdf_lut(float cos_theta, float roughness, float ior)
{
  if (ior <= 1e-5) {
    return vec2(0.0);
  }

  if (ior >= 1.0) {
    vec2 split_sum = brdf_lut(cos_theta, roughness);
    float f0 = f0_from_ior(ior);
    /* Baked IOR for GGX BRDF. */
    const float specular = 1.0;
    const float eta_brdf = (2.0 / (1.0 - sqrt(0.08 * specular))) - 1.0;
    /* Avoid harsh transition comming from ior == 1. */
    float f90 = fast_sqrt(saturate(f0 / (f0_from_ior(eta_brdf) * 0.25)));
    float fresnel = F_brdf_single_scatter(vec3(f0), vec3(f90), split_sum).r;
    /* Setting the BTDF to one is not really important since it is only used for multiscatter
     * and it's already quite close to ground truth. */
    float btdf = 1.0;
    return vec2(btdf, fresnel);
  }

  vec3 coords = lut_coords_btdf(cos_theta, roughness, ior);

  float layer = coords.z * UTIL_BTDF_LAYER_COUNT;
  float layer_floored = floor(layer);

  coords.z = UTIL_BTDF_LAYER + layer_floored;
  vec2 btdf_low = utility_tx_sample(coords.xy, coords.z).rg;

  coords.z += 1.0;
  vec2 btdf_high = utility_tx_sample(coords.xy, coords.z).rg;

  /* Manual trilinear interpolation. */
  vec2 btdf = mix(btdf_low, btdf_high, layer - layer_floored);

  return btdf;
}

/** \} */
