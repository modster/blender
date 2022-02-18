
/**
 * Generate camera-facing quad procedurally for each reflection cubemap of the lightcache.
 */

#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)

void main(void)
{
  const vec2 pos[6] = array(vec2)(vec2(-1.0, -1.0),
                                  vec2(1.0, -1.0),
                                  vec2(-1.0, 1.0),

                                  vec2(1.0, -1.0),
                                  vec2(1.0, 1.0),
                                  vec2(-1.0, 1.0));

  interp.samp = 1 + (gl_VertexID / 6);
  interp.coord = pos[gl_VertexID % 6];

  CubemapData cube = cubes_buf[interp.samp];

  vec3 quad = vec3(interp.coord * probes_buf.cubes_info.display_size * 0.5, 0.0);

  interp.P = vec3(cube._world_position_x, cube._world_position_y, cube._world_position_z);
  interp.P += transform_direction(ViewMatrixInverse, quad);

  gl_Position = point_world_to_ndc(interp.P);
}
