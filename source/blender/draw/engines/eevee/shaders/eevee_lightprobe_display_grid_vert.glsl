
/**
 * Generate camera-facing quad procedurally for each irradiance sample of the lightcache.
 */

#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_lightprobe_display_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)

layout(std140) uniform grids_block
{
  GridData grids[GRID_MAX];
};

layout(std140) uniform lightprobes_info_block
{
  LightProbeInfoData probes_info;
};

uniform int grid_id;

const vec2 pos[6] = vec2[6](vec2(-1.0, -1.0),
                            vec2(1.0, -1.0),
                            vec2(-1.0, 1.0),

                            vec2(1.0, -1.0),
                            vec2(1.0, 1.0),
                            vec2(-1.0, 1.0));

void main(void)
{

  interp.sample = gl_VertexID / 6;
  interp.coord = pos[gl_VertexID % 6];

  GridData grid = grids[grid_id];

  ivec3 cell_coord;
  cell_coord.x = (interp.sample % grid.resolution.x);
  cell_coord.y = (interp.sample / grid.resolution.x) % grid.resolution.y;
  cell_coord.z = (interp.sample / grid.resolution.x) / grid.resolution.y;

  interp.sample += grid.offset;

  mat4 cell_to_world = mat4(vec4(grid.increment_x, 0.0),
                            vec4(grid.increment_y, 0.0),
                            vec4(grid.increment_z, 0.0),
                            vec4(grid.corner, 1.0));

  vec3 quad = vec3(interp.coord * probes_info.grids.display_size * 0.5, 0.0);

  interp.P = transform_point(cell_to_world, vec3(cell_coord));
  interp.P += transform_direction(ViewMatrixInverse, quad);

  gl_Position = point_world_to_ndc(interp.P);
}
