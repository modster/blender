
/**
 * Virtual shadowmapping: Depth buffer scanning.
 * We iterate through the visible lights at each scene pixel depth in order to tag only the visible
 * shadow pages.
 */

#pragma BLENDER_REQUIRE(common_intersection_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_geom_lib.glsl)
#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_culling_iter_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_light_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shadow_lib.glsl)

void tag_tilemap(uint l_idx, vec3 P, float dist_to_cam, const bool is_directional)
{
  LightData light = lights_buf[l_idx];
  ShadowData shadow = light.shadow_data;

  if (light.shadow_id == LIGHT_NO_SHADOW) {
    return;
  }

  int lod = 0;
  ivec2 tile_co;
  int tilemap_index = shadow.tilemap_index;
  if (is_directional) {
    int clipmap_lod = shadow_directional_clipmap_level(shadow, dist_to_cam);
    int clipmap_lod_relative = clipmap_lod - shadow.clipmap_lod_min;
    /* Compute how many time we need to subdivide. */
    float clipmap_res_mul = float(1 << (shadow.clipmap_lod_max - clipmap_lod));
    /* Compute offset of the clipmap from the largest LOD. */
    vec2 clipmap_offset = vec2(abs(shadow.base_offset) >> clipmap_lod_relative) *
                          sign(shadow.base_offset);

    /* [-SHADOW_TILEMAP_RES/2..SHADOW_TILEMAP_RES/2] range for highest LOD. */
    vec3 lP = transform_point(shadow.mat, P);
    tile_co = ivec2(floor(lP.xy * clipmap_res_mul - clipmap_offset)) + SHADOW_TILEMAP_RES / 2;
    tile_co = clamp(tile_co, ivec2(0), ivec2(SHADOW_TILEMAP_RES - 1));
    tilemap_index += clipmap_lod_relative;
    tilemap_index = clamp(tilemap_index, shadow.tilemap_index, shadow.tilemap_last);
  }
  else {
    vec3 lL = light_world_to_local(light, P - light._position);
    float dist_to_light = length(lL);
    if (dist_to_light > light.influence_radius_max) {
      return;
    }
    /* How much a shadow map pixel covers a final image pixel. */
    float footprint_ratio = dist_to_light * (tilemap_pixel_radius * screen_pixel_radius_inv);
    /* Project the radius to the screen. 1 unit away from the camera the same way
     * pixel_world_radius_inv was computed. Not needed in orthographic mode. */
    bool is_persp = (ProjectionMatrix[3][3] == 0.0);
    if (is_persp) {
      footprint_ratio /= dist_to_cam;
    }
    lod = int(ceil(-log2(footprint_ratio)));
    lod = clamp(lod, 0, SHADOW_TILEMAP_LOD);

    int face_id = shadow_punctual_face_index_get(lL);
    lL = shadow_punctual_local_position_to_face_local(face_id, lL);

    uint lod_res = uint(SHADOW_TILEMAP_RES) >> uint(lod);
    tile_co = ivec2(((lL.xy / abs(lL.z)) * 0.5 + 0.5) * float(lod_res));
    tile_co = clamp(tile_co, ivec2(0), ivec2(lod_res - 1));
    tilemap_index += face_id;
  }

  const uint flag = SHADOW_TILE_IS_USED;
  shadow_tile_set_flag(tilemaps_img, tile_co, lod, tilemap_index, flag);
}

void main()
{
  ivec2 texel = ivec2(gl_GlobalInvocationID.xy);
  ivec2 tex_size = textureSize(depth_tx, 0).xy;

  if (!in_range_inclusive(texel, ivec2(0), ivec2(tex_size - 1))) {
    return;
  }

  float depth = texelFetch(depth_tx, texel, 0).r;
  vec2 uv = vec2(texel) / vec2(tex_size);
  vec3 vP = get_view_space_from_depth(uv, depth);
  vec3 P = transform_point(ViewMatrixInverse, vP);

  if (depth == 1.0) {
    return;
  }

  float dist_to_cam = length(vP);

  LIGHT_FOREACH_BEGIN_DIRECTIONAL (lights_cull_buf, l_idx) {
    tag_tilemap(l_idx, P, dist_to_cam, true);
  }
  LIGHT_FOREACH_END

  vec2 px = vec2(gl_GlobalInvocationID.xy);
  LIGHT_FOREACH_BEGIN_LOCAL (lights_cull_buf, lights_zbin_buf, lights_tile_buf, px, vP.z, l_idx) {
    tag_tilemap(l_idx, P, dist_to_cam, false);
  }
  LIGHT_FOREACH_END
}