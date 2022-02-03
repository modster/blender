
/**
 * Generate camera-facing quad procedurally for each irradiance sample of the lightcache.
 */

#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)

void main(void)
{
  const vec2 pos[6] = vec2[6](vec2(-1.0, -1.0),
                              vec2(1.0, -1.0),
                              vec2(-1.0, 1.0),

                              vec2(1.0, -1.0),
                              vec2(1.0, 1.0),
                              vec2(-1.0, 1.0));

  interp.samp = gl_VertexID / 6;
  interp.coord = pos[gl_VertexID % 6];

  GridData grid = grids_buf[grid_id];

  ivec3 cell_coord = grid_cell_index_to_coordinate(interp.samp, grid.resolution);

  interp.samp += grid.offset;

  mat4 cell_to_world = mat4(vec4(grid.increment_x, 0.0),
                            vec4(grid.increment_y, 0.0),
                            vec4(grid.increment_z, 0.0),
                            vec4(grid.corner, 1.0));

  vec3 quad = vec3(interp.coord * probes_buf.grids_info.display_size * 0.5, 0.0);

  interp.P = transform_point(cell_to_world, vec3(cell_coord));
  interp.P += transform_direction(ViewMatrixInverse, quad);

  gl_Position = point_world_to_ndc(interp.P);
}
