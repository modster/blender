
#pragma BLENDER_REQUIRE(common_math_lib.glsl)

IN_OUT MeshDataInterface
{
  vec3 P;
  vec3 N;
}
interp;

struct MeshData {
  /** World position. */
  vec3 P;
  /** Surface Normal. */
  vec3 N;
  /** Geometric Normal. */
  vec3 Ng;
  /** Barycentric coordinates. */
  vec2 barycentrics;
};

#ifdef GPU_FRAGMENT_SHADER
MeshData init_from_interp(void)
{
  MeshData surf;
  surf.P = interp.P;
  surf.N = normalize(interp.N);
  surf.Ng = safe_normalize(cross(dFdx(surf.P), dFdy(surf.P)));
  return surf;
}
#endif