
#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_hair_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_surface_lib.glsl)

void main(void)
{
  interp.P = point_object_to_world(pos);
  interp.N = normal_object_to_world(nor);

  gl_Position = point_world_to_ndc(interp.P);
}