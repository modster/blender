
/**
 * This is an eval function that needs to be added after main fragment shader.
 * A prototype needs to be declared before main in order to use it.
 *
 * The resources expected to be defined are:
 * - probes_info
 * - lightprobe_cube_tx
 * - cubes
 *
 * All of this is needed to avoid using macros and performance issues with large
 * arrays as function arguments.
 */

#pragma BLENDER_REQUIRE(common_math_geom_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_cubemap_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)

float lightprobe_cubemap_weight(CubemapData cube, vec3 P)
{
  /* Composed transform to remove the packed data. */
  vec3 lP = transform_direction(cube.influence_mat, P) + cube.influence_mat[3].xyz;
  float attenuation;
  if (cube._attenuation_type == CUBEMAP_SHAPE_SPHERE) {
    attenuation = saturate(cube._attenuation_factor * (1.0 - length(lP)));
  }
  else {
    attenuation = min_v3(saturate(cube._attenuation_factor * (1.0 - abs(lP))));
  }
  return attenuation;
}

vec3 lightprobe_cubemap_evaluate(CubemapInfoData info,
                                 samplerCubeArray cubemap_tx,
                                 CubemapData cube,
                                 vec3 P,
                                 vec3 R,
                                 float roughness)
{
  float linear_roughness = fast_sqrt(roughness);
  if (cube._layer > 0.0) {
    /* Correct reflection ray using parallax volume intersection. */
    vec3 lR = transform_direction(cube.parallax_mat, R);
    /* Composed transform to remove the packed data. */
    vec3 lP = transform_direction(cube.parallax_mat, P) + cube.parallax_mat[3].xyz;
    float dist;
    if (cube._parallax_type == CUBEMAP_SHAPE_SPHERE) {
      dist = line_unit_sphere_intersect_dist(lP, lR);
    }
    else {
      dist = line_unit_box_intersect_dist(lP, lR);
    }
    vec3 cube_pos = vec3(cube._world_position_x, cube._world_position_y, cube._world_position_z);
    /* Use Distance in WS directly to recover intersection. */
    vec3 intersection = (P + R * dist) - cube_pos;
    /* Distance based roughness from Frostbite PBR Course.
     * http://www.frostbite.com/wp-content/uploads/2014/11/course_notes_moving_frostbite_to_pbr.pdf
     */
    float original_roughness = roughness;
    float distance_roughness = saturate(dist * linear_roughness / length(intersection));
    linear_roughness = mix(distance_roughness, linear_roughness, linear_roughness);
    roughness = linear_roughness * linear_roughness;

    float fac = saturate(original_roughness * 2.0 - 1.0);
    R = mix(intersection, R, fac * fac);
  }
  else {
    R = transform_direction(info.lookdev_rotation, R);
  }
  float lod = linear_roughness * info.roughness_max_lod;
  return cubemap_array_sample(cubemap_tx, vec4(R, cube._layer), lod).rgb;
}

vec3 lightprobe_cubemap_eval(ClosureReflection reflection, vec3 P, vec3 R, float random_threshold)
{
  /* Go through all cubemaps, computing and adding their weights for this pixel
   * until reaching a random threshold. */
  float weight = 0.0;
  int cube_index = probes_info.cubes.cube_count - 1;
  for (; cube_index > 0; cube_index--) {
    weight += lightprobe_cubemap_weight(cubes[cube_index], P);
    if (weight >= random_threshold) {
      break;
    }
  }

  return lightprobe_cubemap_evaluate(
      probes_info.cubes, lightprobe_cube_tx, cubes[cube_index], P, R, reflection.roughness);
}
