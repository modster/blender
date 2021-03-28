
#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(mesh_lib.glsl)

layout(location = 0) out vec4 color;

MeshData surf;

void main(void)
{
  surf.P = interp.P;
  surf.N = normalize(interp.N);
  surf.Ng = safe_normalize(cross(dFdx(surf.P), dFdy(surf.P)));

  color.a = 1.0;
  color.rgb = vec3(saturate(surf.N.z));
}