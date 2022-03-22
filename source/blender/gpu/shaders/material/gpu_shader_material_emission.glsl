void node_emission(vec4 color, float strength, float weight, out Closure result)
{
#ifdef GPU_NODES_SAMPLE_BSDF
  g_emission_data.emission += color.rgb * strength * weight;
#else
  ClosureEmission emission_data = ClosureEmission(color.rgb * strength * weight);
  result = closure_inline_eval(emission_data);
#endif
}
