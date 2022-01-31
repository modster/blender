
#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_cubemap_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_lightprobe_display_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)

uniform samplerCubeArray lightprobe_cube_tx;

layout(location = 0) out vec4 out_color;

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

  out_color.rgb = cubemap_array_sample(lightprobe_cube_tx, vec4(R, interp.sample), 0.0).rgb;
  out_color.a = 0.0;
}
