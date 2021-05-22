
void node_bsdf_translucent(vec4 color, vec3 N, float weight, out Closure result)
{
  closure_weight_add(g_diffuse_data, weight);
}

void node_bsdf_translucent_eval(vec4 color, vec3 N, float weight, out Closure result)
{
  if (closure_weight_threshold(g_diffuse_data, weight)) {
    g_diffuse_data.color = color.rgb * weight;
    g_diffuse_data.N = -N;
  }
}
