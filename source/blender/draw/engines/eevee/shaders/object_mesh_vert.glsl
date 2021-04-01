
#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(object_lib.glsl)

in vec3 pos;
in vec3 nor;

void main(void)
{
  interp.P = point_object_to_world(pos);
  interp.N = normal_object_to_world(nor);

  gl_Position = point_world_to_ndc(interp.P);
}