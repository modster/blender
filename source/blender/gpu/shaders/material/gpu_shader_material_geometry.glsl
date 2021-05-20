void node_geometry(vec3 orco,
                   out vec3 position,
                   out vec3 normal,
                   out vec3 tangent,
                   out vec3 true_normal,
                   out vec3 incoming,
                   out vec3 parametric,
                   out float backfacing,
                   out float pointiness,
                   out float random_per_island)
{
  /* handle perspective/orthographic */
  incoming = cameraVec(g_data.P);

#if defined(WORLD_BACKGROUND) || defined(PROBE_CAPTURE)
  position = -incoming;
  true_normal = normal = incoming;
  tangent = parametric = vec3(0.0);
  vec3(0.0);
  backfacing = 0.0;
  pointiness = 0.0;
#else

  position = g_data.P;
#  ifndef VOLUMETRICS
  normal = normalize(g_data.N);
  vec3 B = dFdx(g_data.P);
  vec3 T = dFdy(g_data.P);
  true_normal = normalize(cross(B, T));
#  else
  normal = (toworld * vec4(g_data.N, 0.0)).xyz;
  true_normal = normal;
#  endif
  tangent_orco_z(orco, orco);
  node_tangent(orco, tangent);

  parametric = vec3(g_data.barycentric_coords, 0.0);
  backfacing = (FrontFacing) ? 0.0 : 1.0;
  pointiness = 0.5;
  random_per_island = 0.0;
#endif
}
