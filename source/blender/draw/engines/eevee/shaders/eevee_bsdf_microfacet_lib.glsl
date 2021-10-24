
/* -------------------------------------------------------------------- */
/** \name GGX
 *
 * \{ */

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
  float a2 = sqr(roughness);

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

float btdf_ggx(vec3 N, vec3 L, vec3 V, float roughness, float eta)
{
  float a2 = sqr(roughness);

  vec3 H = normalize(L + V);
  float NH = max(dot(N, H), 1e-8);
  float NL = max(dot(N, L), 1e-8);
  float NV = max(dot(N, V), 1e-8);
  float VH = max(dot(V, H), 1e-8);
  float LH = max(dot(L, H), 1e-8);
  float Ht2 = sqr(eta * LH + VH);

  float G = G1_Smith_GGX_opti(NV, a2) * G1_Smith_GGX_opti(NL, a2); /* Doing RCP at the end */
  float D = D_ggx_opti(NH, a2);

  /* btdf = abs(VH*LH) * (ior*ior) * D * G(V) * G(L) / (Ht2 * NV) */
  return abs(VH * LH) * sqr(eta) * 4.0 * a2 / (D * G * (Ht2 * NV));
}

/** \} */
