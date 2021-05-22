
void node_bsdf_glossy(
    vec4 color, float roughness, vec3 N, float weight, float use_multiscatter, out Closure result)
{
  g_reflection_data.color = color.rgb;
  g_reflection_data.N = N;
  g_reflection_data.roughness = roughness;
}
