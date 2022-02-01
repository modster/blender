
#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_irradiance_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)

void main()
{
  float len_sqr = len_squared(interp.coord);
  if (len_sqr > 1.0) {
    discard;
  }

  vec3 vN = vec3(interp.coord, sqrt(1.0 - len_sqr));
  vec3 N = normal_view_to_world(vN);

  out_color.rgb = irradiance_load_cell(probes_info.grids, lightprobe_grid_tx, interp.sample, N);
  out_color.a = 0.0;
}
