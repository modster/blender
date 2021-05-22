
void node_bsdf_translucent(vec4 color, vec3 N, float weight,out Closure result)
{
  g_diffuse_data.color = color.rgb;
  g_diffuse_data.N = -N;
}
