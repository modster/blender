
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

#if defined(GPU_FRAGMENT_SHADER) || defined(GPU_VERTEX_SHADER)
GlobalData init_globals(void)
{
  GlobalData surf;
  surf.P = interp.P;
  surf.N = normalize(interp.N);
#  ifndef MAT_GEOM_GPENCIL
  surf.N = (FrontFacing) ? surf.N : -surf.N;
#  endif
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
