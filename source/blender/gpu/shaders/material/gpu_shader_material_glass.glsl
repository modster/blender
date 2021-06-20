
void node_bsdf_glass(vec4 color,
                     float roughness,
                     float ior,
                     vec3 N,
                     float weight,
                     float do_multiscatter,
                     out Closure result,
                     out float reflection_weight,
                     out float refraction_weight)
{
  N = safe_normalize(N);
  vec3 V = cameraVec(g_data.P);
  float NV = dot(N, V);

  float fresnel = (do_multiscatter != 0.0) ? btdf_lut(NV, roughness, ior).y : F_eta(ior, NV);
  reflection_weight = fresnel;
  refraction_weight = 1.0 - fresnel;
  closure_weight_add(g_reflection_data, reflection_weight);
  closure_weight_add(g_refraction_data, refraction_weight);
}

void node_bsdf_glass_eval(vec4 color,
                          float roughness,
                          float ior,
                          vec3 N,
                          float weight,
                          float do_multiscatter,
                          float reflection_weight,
                          float refraction_weight,
                          out Closure result)
{
  N = safe_normalize(N);
  if (closure_weight_threshold(g_reflection_data, reflection_weight)) {
    g_reflection_data.color = color.rgb * reflection_weight;
    g_reflection_data.N = N;
    g_reflection_data.roughness = roughness;
  }
  if (closure_weight_threshold(g_refraction_data, refraction_weight)) {
    vec3 V = cameraVec(g_data.P);
    float NV = dot(N, V);
    float btdf = (do_multiscatter != 0.0) ? 1.0 : btdf_lut(NV, roughness, ior).x;

    g_refraction_data.color = color.rgb * (refraction_weight * btdf);
    g_refraction_data.N = N;
    g_refraction_data.roughness = roughness;
    g_refraction_data.ior = ior;
  }
}
