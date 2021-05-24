
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(gpu_shader_codegen_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_nodetree_eval_lib.glsl)

IN_OUT SurfaceInterface
{
  vec3 P;
  vec3 N;
  vec2 barycentric_coords;
  flat vec3 barycentric_dists;
}
interp;

#ifdef GPU_GEOMETRY_SHADER
vec3 calc_barycentric_distances(vec3 pos0, vec3 pos1, vec3 pos2)
{
  vec3 edge21 = pos2 - pos1;
  vec3 edge10 = pos1 - pos0;
  vec3 edge02 = pos0 - pos2;
  vec3 d21 = normalize(edge21);
  vec3 d10 = normalize(edge10);
  vec3 d02 = normalize(edge02);

  vec3 dists;
  float d = dot(d21, edge02);
  dists.x = sqrt(dot(edge02, edge02) - d * d);
  d = dot(d02, edge10);
  dists.y = sqrt(dot(edge10, edge10) - d * d);
  d = dot(d10, edge21);
  dists.z = sqrt(dot(edge21, edge21) - d * d);
  return dists;
}

vec2 calc_barycentric_co(int vertid)
{
  return vec2((vertid % 3) == 0, (vertid % 3) == 1);
}
#endif

#if defined(GPU_FRAGMENT_SHADER) || defined(GPU_VERTEX_SHADER)
GlobalData init_globals(void)
{
  GlobalData surf;
  surf.P = interp.P;
  surf.N = normalize(interp.N);
#  ifdef GPU_FRAGMENT_SHADER
  surf.Ng = safe_normalize(cross(dFdx(surf.P), dFdy(surf.P)));
#  else
  surf.Ng = surf.N;
#  endif
  surf.barycentric_coords = interp.barycentric_coords;
  surf.barycentric_dists = interp.barycentric_dists;
  surf.ray_type = RAY_TYPE_CAMERA;
  surf.ray_depth = 0.0;
  surf.ray_length = distance(surf.P, cameraPos);
  surf.closure_rand = 0.5;
  return surf;
}
#endif
