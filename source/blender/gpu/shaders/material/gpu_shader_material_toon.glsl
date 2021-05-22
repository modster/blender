void node_bsdf_toon(
    vec4 color, float size, float tsmooth, vec3 N, float weight, out Closure result)
{
  closure_weight_add(g_diffuse_data, weight);
}

void node_bsdf_toon_eval(vec4 color, float roughness, vec3 N, float weight, out Closure result)
{
  if (closure_weight_threshold(g_diffuse_data, weight)) {
    g_diffuse_data.color = color.rgb * weight;
    g_diffuse_data.N = N;
  }
}
