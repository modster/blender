
#pragma BLENDER_REQUIRE(common_math_lib.glsl)

IN_OUT SurfaceInterface
{
  vec3 P;
  vec3 N;
}
interp;

#ifdef GPU_FRAGMENT_SHADER
GlobalData init_from_interp(void)
{
  GlobalData surf;
  surf.P = interp.P;
  surf.N = normalize(interp.N);
  surf.Ng = safe_normalize(cross(dFdx(surf.P), dFdy(surf.P)));
  return surf;
}
#endif