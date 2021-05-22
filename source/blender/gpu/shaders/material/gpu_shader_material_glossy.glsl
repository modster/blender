
void node_bsdf_glossy(vec4 color, float roughness, vec3 N, float weight, out Closure result)
{
  closure_weight_add(g_reflection_data, weight);
}

void node_bsdf_glossy_eval(
    vec4 color, float roughness, vec3 N, float weight, float use_multiscatter, out Closure result)
{
  if (closure_weight_threshold(g_reflection_data, weight)) {
    g_reflection_data.color = color.rgb * weight;
    g_reflection_data.N = N;
    g_reflection_data.roughness = roughness;
  }
}
