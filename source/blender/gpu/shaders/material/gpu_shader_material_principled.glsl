
vec3 tint_from_color(vec3 color)
{
  float lum = dot(color, vec3(0.3, 0.6, 0.1));  /* luminance approx. */
  return (lum > 0.0) ? color / lum : vec3(1.0); /* normalize lum. to isolate hue+sat */
}

float principled_sheen(float NV)
{
  float f = 1.0 - NV;
  /* Empirical approximation (manual curve fitting). Can be refined. */
  float sheen = f * f * f * 0.077 + f * 0.01 + 0.00026;
  return sheen;
}

void node_bsdf_principled(vec4 base_color,
                          float subsurface,
                          vec3 subsurface_radius,
                          vec4 subsurface_color,
                          float metallic,
                          float specular,
                          float specular_tint,
                          float roughness,
                          float anisotropic,
                          float anisotropic_rotation,
                          float sheen,
                          float sheen_tint,
                          float clearcoat,
                          float clearcoat_roughness,
                          float ior,
                          float transmission,
                          float transmission_roughness,
                          vec4 emission,
                          float emission_strength,
                          float alpha,
                          vec3 N,
                          vec3 CN,
                          vec3 T,
                          const float do_diffuse,
                          const float do_clearcoat,
                          const float do_refraction,
                          const float do_multiscatter,
                          out Closure result)
{
  /* Match cycles. */
  metallic = saturate(metallic);
  transmission = saturate(transmission);
  float diffuse_weight = (1.0 - transmission) * (1.0 - metallic);
  transmission *= (1.0 - metallic);
  float specular_weight = (1.0 - transmission);
  clearcoat = max(clearcoat, 0.0);
  transmission_roughness = 1.0 - (1.0 - roughness) * (1.0 - transmission_roughness);
  specular = max(0.0, specular);

  vec3 base_color_tint = tint_from_color(base_color.rgb);

  N = safe_normalize(N);
  vec3 V = cameraVec(g_data.P);
  float NV = dot(N, V);

  if (diffuse_weight > 1e-5) {
    g_diffuse_data.color = mix(base_color.rgb, subsurface_color.rgb, subsurface);
    /* Sheen Coarse approximation: We reuse the diffuse radiance and just scale it. */
    vec3 sheen_color = mix(vec3(1.0), base_color_tint, sheen_tint);
    g_diffuse_data.color += sheen * sheen_color * principled_sheen(NV);
    g_diffuse_data.color *= diffuse_weight;
    g_diffuse_data.N = N;
    g_diffuse_data.sss_radius = subsurface_radius * subsurface;
  }

  // float fresnel = (do_multiscatter != 0.0) ? btdf_lut(NV, roughness, ior).y : F_eta(ior, NV);
  float fresnel = 0.0;

  // vec2 split_sum = brdf_lut(NV, roughness);

  /* TODO(fclem) sample clearcoat, specular or glass reflection layer randomly. */
  if (transmission > 1e-5 && false) {
    /* Poor approximation since we baked the LUT using a fixed IOR. */
    vec3 f0 = mix(vec3(1.0), base_color.rgb, specular_tint);
    vec3 f90 = vec3(1);

    // vec3 brdf = (do_multiscatter != 0.0) ? F_brdf_multi_scatter(f0, f90, split_sum) :
    //                                        F_brdf_single_scatter(f0, f90, split_sum);
    vec3 brdf = f0;

    g_reflection_data.color = brdf * fresnel * transmission;
    g_reflection_data.N = N;
  }
  else if (true) {
    vec3 dielectric_f0_color = mix(vec3(1.0), base_color_tint, specular_tint);
    vec3 metallic_f0_color = base_color.rgb;
    vec3 f0 = mix((0.08 * specular) * dielectric_f0_color, metallic_f0_color, metallic);
    /* Cycles does this blending using the microfacet fresnel factor. However, our fresnel
     * is already baked inside the split sum LUT. We approximate using by modifying the
     * changing the f90 color directly in a non linear fashion. */
    vec3 f90 = mix(f0, vec3(1), fast_sqrt(specular));

    // vec3 brdf = (do_multiscatter != 0.0) ? F_brdf_multi_scatter(f0, f90, split_sum) :
    //                                        F_brdf_single_scatter(f0, f90, split_sum);
    vec3 brdf = f0;

    g_reflection_data.color = brdf * specular_weight;
    g_reflection_data.N = N;
    g_reflection_data.roughness = roughness;
  }
  else {
    N = safe_normalize(CN);
    // float NV = dot(CN, V);
    // vec2 split_sum = brdf_lut(NV, roughness);
    // vec3 brdf = F_brdf_single_scatter(vec3(0.04), vec3(1.0), split_sum);
    vec3 brdf = vec3(1.0);

    g_reflection_data.color = brdf * clearcoat * 0.25;
    g_reflection_data.N = CN;
    g_reflection_data.roughness = clearcoat_roughness;
  }

  if (transmission > 1e-5) {
    // float btdf = (do_multiscatter != 0.0) ?
    //                  1.0 :
    //                  btdf_lut(NV, in_Refraction_3.roughness, in_Refraction_3.ior).x;
    float btdf = 1.0;

    g_refraction_data.color = base_color.rgb * (btdf * (1.0 - fresnel) * transmission);
    g_refraction_data.N = N;
    g_refraction_data.roughness = do_multiscatter != 0.0 ? roughness : transmission_roughness;
  }

  g_emission_data.emission = emission.rgb * emission_strength;

  g_diffuse_data.color *= alpha;
  g_reflection_data.color *= alpha;
  g_refraction_data.color *= alpha;
  g_transparency_data.transmittance = vec3(1.0 - alpha);
}
