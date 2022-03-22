
void node_bsdf_translucent_eval(vec4 color, vec3 N, float weight, out Closure result)
{
#ifdef GPU_NODES_SAMPLE_BSDF
/* TODO(@fclem): Real translucent closure. */
#  define translucent_data g_diffuse_data
#else
  ClosureDiffuse translucent_data;
#endif

  if (closure_weight_threshold(translucent_data, weight)) {
    translucent_data.color = color.rgb * weight;
    translucent_data.N = -N;
    translucent_data.sss_id = 0u;
  }

#ifndef GPU_NODES_SAMPLE_BSDF
  result = closure_inline_eval(translucent_data);
#endif
}

void node_bsdf_translucent(vec4 color, vec3 N, float weight, out Closure result)
{
#ifdef GPU_NODES_SAMPLE_BSDF
  closure_weight_add(g_diffuse_data, weight);
#else
  node_bsdf_translucent_eval(color, roughness, N, weight, result);
#endif
}
