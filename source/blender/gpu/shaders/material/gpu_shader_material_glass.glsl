
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
#ifdef GPU_NODES_SAMPLE_BSDF
#  define reflection_data g_reflection_data
#  define refraction_data g_refraction_data
#else
  ClosureReflection reflection_data;
  ClosureRefraction refraction_data;
#endif

  N = safe_normalize(N);
  if (closure_weight_threshold(reflection_data, reflection_weight)) {
    reflection_data.color = color.rgb * reflection_weight;
    reflection_data.N = N;
    reflection_data.roughness = roughness;
  }
  if (closure_weight_threshold(refraction_data, refraction_weight)) {
    vec3 V = cameraVec(g_data.P);
    float NV = dot(N, V);
    float btdf = (do_multiscatter != 0.0) ? 1.0 : btdf_lut(NV, roughness, ior).x;

    refraction_data.color = color.rgb * (refraction_weight * btdf);
    refraction_data.N = N;
    refraction_data.roughness = roughness;
    refraction_data.ior = ior;
  }

#ifndef GPU_NODES_SAMPLE_BSDF
  result = closure_inline_eval(reflection_data, refraction_data);
#endif
}

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
#ifdef GPU_NODES_SAMPLE_BSDF
  closure_weight_add(g_reflection_data, reflection_weight);
  closure_weight_add(g_refraction_data, refraction_weight);
#else
  node_bsdf_glass_eval(color,
                       roughness,
                       ior,
                       N,
                       weight,
                       do_multiscatter,
                       reflection_weight,
                       refraction_weight,
                       result);
#endif
}
