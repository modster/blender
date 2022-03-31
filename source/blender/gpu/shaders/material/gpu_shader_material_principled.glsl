
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
                               float do_sss,
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

#ifdef GPU_NODES_SAMPLE_BSDF
#  define diffuse_data g_diffuse_data
#  define glass_reflect_data g_reflection_data
#  define reflection_data g_reflection_data
#  define clearcoat_data g_reflection_data
#  define refraction_data g_refraction_data
  /* Emission closure already evaluated. */
  /* Transparency closure already evaluated. */
#else
  ClosureDiffuse diffuse_data;
  ClosureReflection reflection_data;
  ClosureReflection glass_reflect_data;
  ClosureReflection clearcoat_data;
  ClosureRefraction refraction_data;
  ClosureEmission emission_data = ClosureEmission(emission.rgb * emission_strength * alpha *
                                                  weight);
  ClosureTransparency transparency_data = ClosureTransparency(vec3(1.0 - alpha) * weight, 0.0);
#endif

  /* Diffuse. */
  if (closure_weight_threshold(diffuse_data, diffuse_weight)) {
    diffuse_data.color = mix(base_color.rgb, subsurface_color.rgb, subsurface);
    /* Sheen Coarse approximation: We reuse the diffuse radiance and just scale it. */
    vec3 sheen_color = mix(vec3(1.0), base_color_tint, sheen_tint);
    diffuse_data.color += sheen * sheen_color * principled_sheen(NV);
    diffuse_data.color *= diffuse_weight;
    diffuse_data.N = N;
    diffuse_data.sss_radius = subsurface_radius * subsurface;
    diffuse_data.sss_id = uint(do_sss);
  }

  /* Reflection. */
  if (closure_weight_threshold(glass_reflect_data, glass_reflection_weight)) {
    /* Poor approximation since we baked the LUT using a fixed IOR. */
    vec3 f0 = mix(vec3(1.0), base_color.rgb, specular_tint);
    vec3 f90 = vec3(1);

    vec3 brdf = (do_multiscatter != 0.0) ? F_brdf_multi_scatter(f0, f90, split_sum) :
                                           F_brdf_single_scatter(f0, f90, split_sum);

    glass_reflect_data.color = brdf * glass_reflection_weight;
    glass_reflect_data.N = N;
    glass_reflect_data.roughness = roughness;
  }
  if (closure_weight_threshold(reflection_data, specular_weight)) {
    vec3 dielectric_f0_color = mix(vec3(1.0), base_color_tint, specular_tint);
    vec3 metallic_f0_color = base_color.rgb;
    vec3 f0 = mix((0.08 * specular) * dielectric_f0_color, metallic_f0_color, metallic);
    /* Cycles does this blending using the microfacet fresnel factor. However, our fresnel
     * is already baked inside the split sum LUT. We approximate by changing the f90 color directly
     * in a non linear fashion. */
    vec3 f90 = mix(f0, vec3(1.0), fast_sqrt(specular));

    vec3 brdf = (do_multiscatter != 0.0) ? F_brdf_multi_scatter(f0, f90, split_sum) :
                                           F_brdf_single_scatter(f0, f90, split_sum);

    reflection_data.color = brdf * specular_weight;
    reflection_data.N = N;
    reflection_data.roughness = roughness;
  }
  if (closure_weight_threshold(clearcoat_data, clearcoat_weight)) {
    N = safe_normalize(CN);
    float NV = dot(CN, V);
    vec2 split_sum = brdf_lut(NV, roughness);
    vec3 brdf = F_brdf_single_scatter(vec3(0.04), vec3(1.0), split_sum);

    clearcoat_data.color = brdf * clearcoat_weight;
    clearcoat_data.N = CN;
    clearcoat_data.roughness = clearcoat_roughness;
  }

  /* Refraction. */
  if (closure_weight_threshold(refraction_data, glass_transmission_weight)) {
    float btdf = (do_multiscatter != 0.0) ? 1.0 : btdf_lut(NV, roughness, ior).x;

    refraction_data.color = base_color.rgb * (btdf * glass_transmission_weight);
    refraction_data.N = N;
    refraction_data.roughness = do_multiscatter != 0.0 ? roughness :
                                                         max(roughness, transmission_roughness);
    refraction_data.ior = ior;
  }

#ifndef GPU_NODES_SAMPLE_BSDF
  /* Avoid 3 glossy evaluation. */
  reflection_data.color += glass_reflect_data.color;

  result = closure_inline_eval(diffuse_data,
                               reflection_data,
                               clearcoat_data,
                               refraction_data,
                               emission_data,
                               transparency_data);
#endif
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
                          float do_sss,
                          out Closure result,
                          out float diffuse_weight,
                          out float specular_weight,
                          out float glass_reflection_weight,
                          out float glass_transmission_weight,
                          out float clearcoat_weight)
{
  /* Match cycles. */
  metallic = clamp(metallic, 0.0, 1.0);
  transmission = clamp(transmission, 0.0, 1.0);
  diffuse_weight = (1.0 - transmission) * (1.0 - metallic);
  transmission *= (1.0 - metallic);
  specular_weight = (1.0 - transmission);
  clearcoat_weight = max(clearcoat, 0.0) * 0.25;
  transmission_roughness = 1.0 - (1.0 - roughness) * (1.0 - transmission_roughness);
  specular = max(0.0, specular);

  N = safe_normalize(N);
  vec3 V = cameraVec(g_data.P);
  float NV = dot(N, V);

#ifdef GPU_NODES_SAMPLE_BSDF
  /* Evaluate non sampled closures. */
  g_emission_data.emission += emission.rgb * emission_strength * weight;
  g_transparency_data.transmittance += vec3(1.0 - alpha) * weight;
#endif
  /* Apply alpha and weight on sampled closures. */
  alpha *= weight;
  diffuse_weight *= alpha;
  specular_weight *= alpha;
  transmission *= alpha;
  clearcoat_weight *= alpha;

  float fresnel = (do_multiscatter != 0.0) ? btdf_lut(NV, roughness, ior).y : F_eta(ior, NV);
  glass_reflection_weight = fresnel * transmission;
  glass_transmission_weight = (1.0 - fresnel) * transmission;

#ifdef GPU_NODES_SAMPLE_BSDF
  closure_weight_add(g_diffuse_data, diffuse_weight);
  closure_weight_add(g_reflection_data, glass_reflection_weight);
  closure_weight_add(g_reflection_data, specular_weight);
  closure_weight_add(g_reflection_data, clearcoat_weight);
  closure_weight_add(g_refraction_data, glass_transmission_weight);
#else
  node_bsdf_principled_eval(base_color,
                            subsurface,
                            subsurface_radius,
                            subsurface_color,
                            subsurface_ior,
                            subsurface_anisotropy,
                            metallic,
                            specular,
                            specular_tint,
                            roughness,
                            anisotropic,
                            anisotropic_rotation,
                            sheen,
                            sheen_tint,
                            clearcoat,
                            clearcoat_roughness,
                            ior,
                            transmission,
                            transmission_roughness,
                            emission,
                            emission_strength,
                            alpha,
                            N,
                            CN,
                            T,
                            weight,
                            do_multiscatter,
                            do_sss,
                            diffuse_weight,
                            specular_weight,
                            glass_reflection_weight,
                            glass_transmission_weight,
                            clearcoat_weight,
                            result);
#endif
}
