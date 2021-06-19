
/**
 * This is an eval function that needs to be added after main fragment shader.
 * A prototype needs to be declared before main in order to use it.
 *
 * The resources expected to be defined are:
 * - probes_info
 * - lightprobe_grid_tx
 * - grids
 *
 * All of this is needed to avoid using macros and performance issues with large
 * arrays as function arguments.
 */

#pragma BLENDER_REQUIRE(common_math_geom_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_irradiance_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)

float lightprobe_grid_weight(GridData grid, vec3 P)
{
  vec3 lP = transform_point(grid.local_mat, P);
  vec3 pos_to_edge = max(vec3(0.0), abs(lP) - 1.0);
  float fade = length(pos_to_edge);
  return saturate(-fade * grid.attenuation_scale + grid.attenuation_bias);
}

vec3 lightprobe_grid_evaluate(
    GridInfoData info, sampler2DArray grid_tx, GridData grid, vec3 P, vec3 N)
{
  vec3 lP = transform_point(grid.local_mat, P) * 0.5 + 0.5;
  lP = lP * vec3(grid.resolution) - 0.5;

  ivec3 lP_floored = ivec3(floor(lP));
  vec3 trilinear_weight = fract(lP);

  if (grid.offset == 0) {
    N = transform_direction(info.lookdev_rotation, N);
  }

  float weight_accum = 0.0;
  vec3 irradiance_accum = vec3(0.0);
  /* For each neighbor cells */
  for (int i = 0; i < 8; i++) {
    ivec3 offset = ivec3(i, i >> 1, i >> 2) & ivec3(1);
    ivec3 cell_coord = clamp(lP_floored + offset, ivec3(0), grid.resolution - 1);
    /* Skip cells not yet rendered during baking. */
    cell_coord = (cell_coord / grid.level_skip) * grid.level_skip;
    /* Keep in sync with update_irradiance_probe. */
    int cell_id = grid.offset + cell_coord.z + cell_coord.y * grid.resolution.z +
                  cell_coord.x * grid.resolution.z * grid.resolution.y;

    vec3 color = irradiance_load_cell(info, grid_tx, cell_id, N);

    /* We need this because we render probes in world space (so we need light vector in WS).
     * And rendering them in local probe space is too much problem. */
    mat4 cell_to_world = mat4(vec4(grid.increment_x, 0.0),
                              vec4(grid.increment_y, 0.0),
                              vec4(grid.increment_z, 0.0),
                              vec4(grid.corner, 1.0));
    vec3 ws_cell_location = transform_point(cell_to_world, vec3(cell_coord));

    vec3 ws_point_to_cell = ws_cell_location - P;
    float ws_dist_point_to_cell = length(ws_point_to_cell);
    vec3 ws_light = ws_point_to_cell / ws_dist_point_to_cell;

    /* Smooth backface test. */
    float weight = saturate(dot(ws_light, N));
    /* Precomputed visibility. */
    weight *= visibility_load_cell(info,
                                   grid_tx,
                                   cell_id,
                                   ws_light,
                                   ws_dist_point_to_cell,
                                   grid.visibility_bias,
                                   grid.visibility_bleed,
                                   grid.visibility_range);
    /* Smoother transition. */
    weight += info.irradiance_smooth;
    /* Trilinear weights. */
    vec3 trilinear = mix(1.0 - trilinear_weight, trilinear_weight, offset);
    weight *= trilinear.x * trilinear.y * trilinear.z;
    /* Avoid zero weight. */
    weight = max(0.00001, weight);

    weight_accum += weight;
    irradiance_accum += color * weight;
  }
  return irradiance_accum / weight_accum;
}

vec3 lightprobe_grid_eval(ClosureDiffuse diffuse, vec3 P, float random_threshold)
{
  /* Go through all grids, computing and adding their weights for this pixel
   * until reaching a random threshold. */
  float weight = 0.0;
  int grid_index = probes_info.grids.grid_count - 1;
  for (; grid_index > 0; grid_index--) {
    weight += lightprobe_grid_weight(grids[grid_index], P);
    if (weight >= random_threshold) {
      break;
    }
  }

  return lightprobe_grid_evaluate(
      probes_info.grids, lightprobe_grid_tx, grids[grid_index], P, diffuse.N);
}
