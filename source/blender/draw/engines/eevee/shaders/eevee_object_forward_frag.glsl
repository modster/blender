
#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_object_lib.glsl)

layout(location = 0, index = 0) out vec4 outRadiance;
layout(location = 0, index = 1) out vec4 outTransmittance;

MeshData g_surf;
ivec4 g_closure_data[8];

void main(void)
{
  g_surf = init_from_interp();

  /* Prevent precision issues on unit coordinates. */
  vec3 p = (g_surf.P + 0.000001) * 0.999999;
  int xi = int(abs(floor(p.x)));
  int yi = int(abs(floor(p.y)));
  int zi = int(abs(floor(p.z)));
  bool check = ((mod(xi, 2) == mod(yi, 2)) == bool(mod(zi, 2)));

  outRadiance = vec4(vec3(g_surf.N.z * 0.5 + 0.5) * mix(0.5, 0.8, check), 1.0);
  outTransmittance = vec4(0.0, 0.0, 0.0, 1.0);
}