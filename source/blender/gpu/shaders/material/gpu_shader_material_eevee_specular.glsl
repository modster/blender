
void node_eevee_specular(vec4 diffuse,
                         vec4 specular,
                         float roughness,
                         vec4 emissive,
                         float transp,
                         vec3 normal,
                         float clearcoat,
                         float clearcoat_roughness,
                         vec3 clearcoat_normal,
                         float occlusion,
                         float weight,
                         out Closure result)
{

  /* Evaluate non sampled closures. */
  g_emission_data.emission += emissive.rgb * weight;
  g_transparency_data.transmittance += vec3(transp) * weight;

  float alpha = (1.0 - transp) * weight;

  closure_weight_add(g_diffuse_data, alpha);
  closure_weight_add(g_reflection_data, alpha);
  closure_weight_add(g_reflection_data, alpha * clearcoat * 0.25);
}

void node_eevee_specular_eval(vec4 diffuse,
                              vec4 specular,
                              float roughness,
                              vec4 emissive,
                              float transp,
                              vec3 N,
                              float clearcoat,
                              float clearcoat_roughness,
                              vec3 CN,
                              float occlusion,
                              float weight,
                              out Closure result)
{
  N = safe_normalize(N);
  vec3 V = cameraVec(g_data.P);

  float alpha = (1.0 - transp) * weight;

  float diffuse_weight = alpha;
  if (closure_weight_threshold(g_diffuse_data, diffuse_weight)) {
    g_diffuse_data.color = diffuse.rgb * diffuse_weight;
    g_diffuse_data.N = N;
  }

  float specular_weight = alpha;
  if (closure_weight_threshold(g_reflection_data, specular_weight)) {
    float NV = dot(N, V);
    vec2 split_sum = brdf_lut(NV, roughness);
    vec3 brdf = F_brdf_single_scatter(specular.rgb, vec3(1.0), split_sum);

    g_reflection_data.color = specular.rgb * brdf * specular_weight;
    g_reflection_data.N = N;
    g_reflection_data.roughness = roughness;
  }

  float clearcoat_weight = alpha * clearcoat * 0.25;
  if (closure_weight_threshold(g_reflection_data, clearcoat_weight)) {
    CN = safe_normalize(CN);
    float NV = dot(CN, V);
    vec2 split_sum = brdf_lut(NV, clearcoat_roughness);
    vec3 brdf = F_brdf_single_scatter(vec3(0.04), vec3(1.0), split_sum);

    g_reflection_data.color = brdf * clearcoat_weight;
    g_reflection_data.N = CN;
    g_reflection_data.roughness = clearcoat_roughness;
  }
}
