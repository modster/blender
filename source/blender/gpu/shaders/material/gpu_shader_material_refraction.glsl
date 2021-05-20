
void node_bsdf_refraction(vec4 color, float roughness, float ior, vec3 N, out Closure result)
{
  g_refraction_data.color = color.rgb;
  g_refraction_data.N = N;
  g_refraction_data.roughness = roughness;
}
