
void node_bsdf_diffuse_eval(vec4 color, float roughness, vec3 N, float weight, out Closure result)
{
#ifdef GPU_NODES_SAMPLE_BSDF
#  define diffuse_data g_diffuse_data
#else
  ClosureDiffuse diffuse_data;
#endif

  if (closure_weight_threshold(diffuse_data, weight)) {
    diffuse_data.color = color.rgb * weight;
    diffuse_data.N = N;
    diffuse_data.sss_id = 0u;
  }

#ifndef GPU_NODES_SAMPLE_BSDF
  result = closure_inline_eval(diffuse_data);
#endif
}

void node_bsdf_diffuse(vec4 color, float roughness, vec3 N, float weight, out Closure result)
{
#ifdef GPU_NODES_SAMPLE_BSDF
  closure_weight_add(g_diffuse_data, weight);
#else
  node_bsdf_diffuse_eval(color, roughness, N, weight, result);
#endif
}
