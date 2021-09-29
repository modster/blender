
void node_subsurface_scattering(vec4 color,
                                float scale,
                                vec3 radius,
                                float ior,
                                float anisotropy,
                                vec3 N,
                                float weight,
                                out Closure result)
{
  closure_weight_add(g_diffuse_data, weight);
}

void node_subsurface_scattering_eval(vec4 color,
                                     float scale,
                                     vec3 radius,
                                     float ior,
                                     float anisotropy,
                                     vec3 N,
                                     float weight,
                                     out Closure result)
{
  if (closure_weight_threshold(g_diffuse_data, weight)) {
    g_diffuse_data.color = color.rgb * weight;
    g_diffuse_data.N = N;
    g_diffuse_data.sss_radius = radius * scale;
  }
}