
void node_subsurface_scattering(vec4 color,
                                float scale,
                                vec3 radius,
                                float sharpen,
                                float texture_blur,
                                vec3 N,
                                out Closure result)
{
  g_diffuse_data.color = color.rgb;
  g_diffuse_data.N = N;
  g_diffuse_data.sss_radius = radius * scale;
}
