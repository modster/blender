
void node_bsdf_diffuse(vec4 color, float roughness, vec3 N, out Closure result)
{
  g_diffuse_data.color = color.rgb;
  g_diffuse_data.N = N;
}
