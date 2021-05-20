void camera(out vec3 outview, out float outdepth, out float outdist)
{
  outdepth = abs(g_data.P.z);
  outdist = length(g_data.P);
  outview = normalize(-g_data.P);
}
