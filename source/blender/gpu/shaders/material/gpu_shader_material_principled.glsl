
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
                          float subsurface_ior,
                          float subsurface_anisotropy,
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
                          float weight,
                          const float do_multiscatter,
                          out Closure result,
                          out float diffuse_weight,
                          out float specular_weight,
                          out float glass_reflection_weight,
                          out float glass_transmission_weight,
                          out float clearcoat_weight)
{
  /* Match cycles. */
  metallic = saturate(metallic);
  transmission = saturate(transmission);
  diffuse_weight = (1.0 - transmission) * (1.0 - metallic);
  transmission *= (1.0 - metallic);
  specular_weight = (1.0 - transmission);
  clearcoat_weight = max(clearcoat, 0.0) * 0.25;
  transmission_roughness = 1.0 - (1.0 - roughness) * (1.0 - transmission_roughness);
  specular = max(0.0, specular);

  N = safe_normalize(N);
  vec3 V = cameraVec(g_data.P);
  float NV = dot(N, V);

  /* Evaluate non sampled closures. */
  g_emission_data.emission += emission.rgb * emission_strength * weight;
  g_transparency_data.transmittance += vec3(1.0 - alpha) * weight;
  /* Apply alpha and weight on sampled closures. */
  alpha *= weight;
  diffuse_weight *= alpha;
  specular_weight *= alpha;
  transmission *= alpha;
  clearcoat_weight *= alpha;

  float fresnel = (do_multiscatter != 0.0) ? btdf_lut(NV, roughness, ior).y : F_eta(ior, NV);
  glass_reflection_weight = fresnel * transmission;
  glass_transmission_weight = (1.0 - fresnel) * transmission;

  closure_weight_add(g_diffuse_data, diffuse_weight);
  closure_weight_add(g_reflection_data, glass_reflection_weight);
  closure_weight_add(g_reflection_data, specular_weight);
  closure_weight_add(g_reflection_data, clearcoat_weight);
  closure_weight_add(g_refraction_data, glass_transmission_weight);
}

void node_bsdf_principled_eval(vec4 base_color,
                               float subsurface,
                               vec3 subsurface_radius,
                               vec4 subsurface_color,
                               float subsurface_ior,
                               float subsurface_anisotropy,
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
                               float weight,
                               const float do_multiscatter,
                               float diffuse_weight,
                               float specular_weight,
                               float glass_reflection_weight,
                               float glass_transmission_weight,
                               float clearcoat_weight,
                               out Closure result)
{
  vec3 base_color_tint = tint_from_color(base_color.rgb);

  N = safe_normalize(N);
  vec3 V = cameraVec(g_data.P);
  float NV = dot(N, V);

  vec2 split_sum = brdf_lut(NV, roughness);

  /* Diffuse. */
  if (closure_weight_threshold(g_diffuse_data, diffuse_weight)) {
    g_diffuse_data.color = mix(base_color.rgb, subsurface_color.rgb, subsurface);
    /* Sheen Coarse approximation: We reuse the diffuse radiance and just scale it. */
    vec3 sheen_color = mix(vec3(1.0), base_color_tint, sheen_tint);
    g_diffuse_data.color += sheen * sheen_color * principled_sheen(NV);
    g_diffuse_data.color *= diffuse_weight;
    g_diffuse_data.N = N;
    g_diffuse_data.sss_radius = subsurface_radius * subsurface;
    g_diffuse_data.sss_id = uint(resource_handle + 1);
  }

  /* Reflection. */
  if (closure_weight_threshold(g_reflection_data, glass_reflection_weight)) {
    /* Poor approximation since we baked the LUT using a fixed IOR. */
    vec3 f0 = mix(vec3(1.0), base_color.rgb, specular_tint);
    vec3 f90 = vec3(1);

    vec3 brdf = (do_multiscatter != 0.0) ? F_brdf_multi_scatter(f0, f90, split_sum) :
                                           F_brdf_single_scatter(f0, f90, split_sum);

    g_reflection_data.color = brdf * glass_reflection_weight;
    g_reflection_data.N = N;
    g_reflection_data.roughness = roughness;
  }
  else if (closure_weight_threshold(g_reflection_data, specular_weight)) {
    vec3 dielectric_f0_color = mix(vec3(1.0), base_color_tint, specular_tint);
    vec3 metallic_f0_color = base_color.rgb;
    vec3 f0 = mix((0.08 * specular) * dielectric_f0_color, metallic_f0_color, metallic);
    /* Cycles does this blending using the microfacet fresnel factor. However, our fresnel
     * is already baked inside the split sum LUT. We approximate using by modifying the
     * changing the f90 color directly in a non linear fashion. */
    vec3 f90 = mix(f0, vec3(1.0), fast_sqrt(specular));

    vec3 brdf = (do_multiscatter != 0.0) ? F_brdf_multi_scatter(f0, f90, split_sum) :
                                           F_brdf_single_scatter(f0, f90, split_sum);

    g_reflection_data.color = brdf * specular_weight;
    g_reflection_data.N = N;
    g_reflection_data.roughness = roughness;
  }
  else if (closure_weight_threshold(g_reflection_data, clearcoat_weight)) {
    N = safe_normalize(CN);
    float NV = dot(CN, V);
    vec2 split_sum = brdf_lut(NV, roughness);
    vec3 brdf = F_brdf_single_scatter(vec3(0.04), vec3(1.0), split_sum);

    g_reflection_data.color = brdf * clearcoat_weight;
    g_reflection_data.N = CN;
    g_reflection_data.roughness = clearcoat_roughness;
  }

  /* Refraction. */
  if (closure_weight_threshold(g_refraction_data, glass_transmission_weight)) {
    float btdf = (do_multiscatter != 0.0) ? 1.0 : btdf_lut(NV, roughness, ior).x;

    g_refraction_data.color = base_color.rgb * (btdf * glass_transmission_weight);
    g_refraction_data.N = N;
    g_refraction_data.roughness = do_multiscatter != 0.0 ? roughness : transmission_roughness;
    g_refraction_data.ior = ior;
  }
}
