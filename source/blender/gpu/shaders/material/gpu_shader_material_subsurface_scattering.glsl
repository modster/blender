
void node_subsurface_scattering_eval(vec4 color,
                                     float scale,
                                     vec3 radius,
                                     float ior,
                                     float anisotropy,
                                     vec3 N,
                                     float weight,
                                     float do_sss,
                                     out Closure result)
{
#ifdef GPU_NODES_SAMPLE_BSDF
#  define diffuse_data g_diffuse_data
#else
  ClosureDiffuse diffuse_data;
#endif

  if (closure_weight_threshold(diffuse_data, weight)) {
    diffuse_data.color = color.rgb * weight;
    diffuse_data.N = N;
    diffuse_data.sss_radius = radius * scale;
    diffuse_data.sss_id = uint(do_sss);
  }

#ifndef GPU_NODES_SAMPLE_BSDF
  result = closure_inline_eval(diffuse_data);
#endif
}

void node_subsurface_scattering(vec4 color,
                                float scale,
                                vec3 radius,
                                float ior,
                                float anisotropy,
                                vec3 N,
                                float weight,
                                float do_sss,
                                out Closure result)
{
#ifdef GPU_NODES_SAMPLE_BSDF
  closure_weight_add(g_diffuse_data, weight);
#else
  node_subsurface_scattering_eval(
      color, scale, radius, ior, anisotropy, N, weight, do_sss, result);
#endif
}
