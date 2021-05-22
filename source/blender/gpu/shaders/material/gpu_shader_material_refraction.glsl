
void node_bsdf_refraction(
    vec4 color, float roughness, float ior, vec3 N, float weight, out Closure result)
{
  closure_weight_add(g_refraction_data, weight);
}

void node_bsdf_refraction_eval(
    vec4 color, float roughness, float ior, vec3 N, float weight, out Closure result)
{
  if (closure_weight_threshold(g_refraction_data, weight)) {
    g_refraction_data.color = color.rgb * weight;
    g_refraction_data.N = N;
    g_refraction_data.roughness = roughness;
  }
}
