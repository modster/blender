
void node_bsdf_glossy_eval(
    vec4 color, float roughness, vec3 N, float weight, float use_multiscatter, out Closure result)
{
#ifdef GPU_NODES_SAMPLE_BSDF
#  define reflection_data g_reflection_data
#else
  ClosureReflection reflection_data;
#endif

  if (closure_weight_threshold(reflection_data, weight)) {
    /* TODO(fclem): Multiscatter. */
    reflection_data.color = color.rgb * weight;
    reflection_data.N = N;
    reflection_data.roughness = roughness;
  }

#ifndef GPU_NODES_SAMPLE_BSDF
  result = closure_inline_eval(reflection_data);
#endif
}

void node_bsdf_glossy(
    vec4 color, float roughness, vec3 N, float weight, float use_multiscatter, out Closure result)
{
#ifdef GPU_NODES_SAMPLE_BSDF
  closure_weight_add(g_reflection_data, weight);
#else
  node_bsdf_glossy_eval(color, roughness, N, weight, use_multiscatter, result);
#endif
}
