
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
  float lod = linear_roughness * info.roughness_max_lod;
  return cubemap_array_sample(cubemap_tx, vec4(R, cube._layer), lod).rgb;
}

/* Go through all grids, computing and adding their weights for this pixel
 * until reaching a random threshold. */
/* Unfortunately, it has to be a define because a lot of compilers do not optimize array of structs
 * references. */
#define lightprobe_cubemap_select(_cubes_info, _cubes, _P, _random_threshold, _out_index) \
  { \
    float weight = 0.0; \
    for (_out_index = _cubes_info.cube_count - 1; _out_index > 0; _out_index--) { \
      weight += lightprobe_cubemap_weight(_cubes[_out_index], _P); \
      if (weight >= _random_threshold) { \
        break; \
      } \
    } \
  }
