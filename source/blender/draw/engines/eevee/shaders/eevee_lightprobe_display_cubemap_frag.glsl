
#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_cubemap_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)

void main()
{
  float len_sqr = len_squared(interp.coord);
  if (len_sqr > 1.0) {
    discard;
  }

  vec3 vN = vec3(interp.coord, sqrt(1.0 - len_sqr));
  vec3 N = normal_view_to_world(vN);

  vec3 V = cameraVec(interp.P);
  vec3 R = -reflect(V, N);

  out_color.rgb = textureLod(lightprobe_cube_tx, vec4(R, interp.sample), 0.0).rgb;
  out_color.a = 0.0;
}
