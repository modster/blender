void node_bsdf_anisotropic(vec4 color,
                           float roughness,
                           float anisotropy,
                           float rotation,
                           vec3 N,
                           vec3 T,
                           float weight,
                           out Closure result)
{
  closure_weight_add(g_reflection_data, weight);
}

void node_bsdf_anisotropic_eval(
    vec4 color, float roughness, vec3 N, float weight, float use_multiscatter, out Closure result)
{
#ifdef GPU_NODES_SAMPLE_BSDF
#  define reflection_data g_reflection_data
#else
  ClosureReflection reflection_data;
#endif

  if (closure_weight_threshold(g_reflection_data, weight)) {
    /* TODO(fclem): Multiscatter. */
    g_reflection_data.color = color.rgb * weight;
    g_reflection_data.N = N;
    g_reflection_data.roughness = roughness;
  }

#ifndef GPU_NODES_SAMPLE_BSDF
  result = closure_inline_eval(reflection_data);
#endif
}
