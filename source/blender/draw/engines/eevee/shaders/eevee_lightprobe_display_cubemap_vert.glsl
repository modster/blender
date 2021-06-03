
/**
 * Generate camera-facing quad procedurally for each reflection cubemap of the lightcache.
 */

#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_lightprobe_display_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)

layout(std140) uniform cubes_block
{
  CubemapData cubes[CULLING_ITEM_BATCH];
};

layout(std140) uniform lightprobes_info_block
{
  LightProbeInfoData probes_info;
};

const vec2 pos[6] = vec2[6](vec2(-1.0, -1.0),
                            vec2(1.0, -1.0),
                            vec2(-1.0, 1.0),

                            vec2(1.0, -1.0),
                            vec2(1.0, 1.0),
                            vec2(-1.0, 1.0));

void main(void)
{
  interp.sample = 1 + (gl_VertexID / 6);
  interp.coord = pos[gl_VertexID % 6];

  CubemapData cube = cubes[interp.sample];

  vec3 quad = vec3(interp.coord * probes_info.cubes.display_size * 0.5, 0.0);

  interp.P = vec3(cube._world_position_x, cube._world_position_y, cube._world_position_z);
  interp.P += transform_direction(ViewMatrixInverse, quad);

  gl_Position = point_world_to_ndc(interp.P);
}
