void camera(out vec3 outview, out float outdepth, out float outdist)
{
  outdepth = abs(transform_point(g_data.P, ViewMatrix).z);
  outdist = length(g_data.P);
  outview = normalize(-g_data.P);
}
