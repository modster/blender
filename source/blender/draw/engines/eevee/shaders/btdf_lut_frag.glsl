#pragma BLENDER_REQUIRE(common_utiltex_lib.glsl)
#pragma BLENDER_REQUIRE(bsdf_sampling_lib.glsl)

uniform float sampleCount;
uniform float z;

out vec4 FragColor;

void main()
{
  float x = floor(gl_FragCoord.x) / (LUT_SIZE - 1.0);
  float y = floor(gl_FragCoord.y) / (LUT_SIZE - 1.0);

  float ior = sqrt(x);
  ior = clamp(sqrt(ior), 0.0, 0.99995);
  /* ior is sin of critical angle. */
  float critical_cos = sqrt(1.0 - saturate(ior * ior));

  /* Manual fit to range. */
  y = y * 1.45 + 0.05;
  /* Remap for better accuracy. */
  float NV = 1.0 - y * y;
  /* Center LUT around critical angle to always have a sharp cut if roughness is 0. */
  NV += critical_cos;
  NV = clamp(NV, 0.0, 0.9999);

  float a = z * z;
  float a2 = clamp(a * a, 1e-8, 0.9999);

  vec3 V = vec3(sqrt(1.0 - NV * NV), 0.0, NV);

  /* Integrating BTDF */
  float btdf_accum = 0.0;
  float fresnel_accum = 0.0;
  for (float j = 0.0; j < sampleCount; j++) {
    for (float i = 0.0; i < sampleCount; i++) {
      vec3 Xi = (vec3(i, j, 0.0) + 0.5) / sampleCount;
      Xi.yz = vec2(cos(Xi.y * M_2PI), sin(Xi.y * M_2PI));

      /* Microfacet normal. */
      vec3 H = sample_ggx(Xi, a2);

      float VH = dot(V, H);

      /* Check if there is total internal reflections. */
      float fresnel = F_eta(ior, VH);

      fresnel_accum += fresnel;

      float eta = 1.0 / ior;
      if (dot(H, V) < 0.0) {
        H = -H;
        eta = ior;
      }

      vec3 L = refract(-V, H, eta);
      float NL = -L.z;

      if ((NL > 0.0) && (fresnel < 1.0)) {
        float LH = dot(L, H);

        /* Balancing the adjustments made in G1_Smith. */
        float G1_l = NL * 2.0 / G1_Smith_GGX(NL, a2);

        /* btdf = abs(VH*LH) * (ior*ior) * D * G(V) * G(L) / (Ht2 * NV)
         * pdf = (VH * abs(LH)) * (ior*ior) * D * G(V) / (Ht2 * NV) */
        float btdf = G1_l * abs(VH * LH) / (VH * abs(LH));

        btdf_accum += btdf;
      }
    }
  }
  btdf_accum /= sampleCount * sampleCount;
  fresnel_accum /= sampleCount * sampleCount;

  /* There is place to put multiscater result (which is a little bit different still)
   * and / or lobe fitting for better sampling of  */
  FragColor = vec4(btdf_accum, fresnel_accum, 0.0, 1.0);
}
