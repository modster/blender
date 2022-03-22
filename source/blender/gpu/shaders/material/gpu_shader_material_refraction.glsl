
void node_bsdf_refraction(
    vec4 color, float roughness, float ior, vec3 N, float weight, out Closure result)
{
#ifdef GPU_NODES_SAMPLE_BSDF
  closure_weight_add(g_refraction_data, weight);
#endif
}

void node_bsdf_refraction_eval(
    vec4 color, float roughness, float ior, vec3 N, float weight, out Closure result)
{
#ifdef GPU_NODES_SAMPLE_BSDF
#  define refraction_data g_refraction_data
#else
  ClosureRefraction refraction_data;
#endif

  if (closure_weight_threshold(refraction_data, weight)) {
    refraction_data.color = color.rgb * weight;
    refraction_data.N = N;
    refraction_data.roughness = roughness;
    refraction_data.ior = ior;
  }

#ifndef GPU_NODES_SAMPLE_BSDF
  result = closure_inline_eval(refraction_data);
#endif
}
