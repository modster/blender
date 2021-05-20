void node_tex_environment_texco(out vec3 worldvec)
{
#ifdef MESH_SHADER
  worldvec = g_data.P;
#else
  vec3 V = cameraVec(g_data.P);

#  if defined(WORLD_BACKGROUND) || defined(PROBE_CAPTURE)
  worldvec = V;
#  else
  worldvec = mat3(ModelMatrixInverse) * V;
#  endif
#endif
}

void node_tex_environment_equirectangular(vec3 co, out vec3 uv)
{
  vec3 nco = normalize(co);
  uv.x = -atan(nco.y, nco.x) / (2.0 * M_PI) + 0.5;
  uv.y = atan(nco.z, hypot(nco.x, nco.y)) / M_PI + 0.5;
}

void node_tex_environment_mirror_ball(vec3 co, out vec3 uv)
{
  vec3 nco = normalize(co);
  nco.y -= 1.0;

  float div = 2.0 * sqrt(max(-0.5 * nco.y, 0.0));
  nco /= max(1e-8, div);

  uv = 0.5 * nco.xzz + 0.5;
}

void node_tex_environment_empty(vec3 co, out vec4 color)
{
  color = vec4(1.0, 0.0, 1.0, 1.0);
}
