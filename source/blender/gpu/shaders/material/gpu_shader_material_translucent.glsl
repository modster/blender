
void node_bsdf_translucent(vec4 color, vec3 N, out Closure result)
{
  g_diffuse_data.color = color.rgb;
  g_diffuse_data.N = -N;
  g_diffuse_data.thickness = 0.0;
  g_diffuse_data.sss_radius = vec3(0);
  g_diffuse_data.sss_id = 0u;
}
