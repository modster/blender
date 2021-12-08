
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)
#pragma BLENDER_REQUIRE(eevee_shadow_tilemap_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shadow_page_lib.glsl)

/* ---------------------------------------------------------------------- */
/** \name Shadow Sampling Functions
 * \{ */

/* Turns local light coordinate into shadow region index. Matches eCubeFace order. */
int shadow_punctual_face_index_get(vec3 lL)
{
  vec3 aP = abs(lL);
  if (all(greaterThan(aP.xx, aP.yz))) {
    return (lL.x > 0.0) ? 1 : 2;
  }
  else if (all(greaterThan(aP.yy, aP.xz))) {
    return (lL.y > 0.0) ? 3 : 4;
  }
  else {
    return (lL.z > 0.0) ? 5 : 0;
  }
}

/* Transform vector to face local coordinate. */
vec3 shadow_punctual_local_position_to_face_local(int face_id, vec3 lL)
{
  switch (face_id) {
    case 1:
      return vec3(-lL.y, lL.z, -lL.x);
    case 2:
      return vec3(lL.y, lL.z, lL.x);
    case 3:
      return vec3(lL.x, lL.z, -lL.y);
    case 4:
      return vec3(-lL.x, lL.z, lL.y);
    case 5:
      return vec3(lL.x, -lL.y, -lL.z);
    default:
      return lL;
  }
}

float shadow_punctual_depth_get(
    sampler2D atlas_tx, usampler2D tilemaps_tx, LightData light, ShadowData shadow, vec3 lL)
{
  lL -= shadow.offset;
  int face_id = shadow_punctual_face_index_get(lL);
  lL = shadow_punctual_local_position_to_face_local(face_id, lL);
  /* UVs in [0..SHADOW_TILEMAP_RES] range. */
  const float lod0_res = float(SHADOW_TILEMAP_RES / 2);
  vec2 uv = (lL.xy / abs(lL.z)) * lod0_res + lod0_res;
  ivec2 tile_co = ivec2(floor(uv));
  int tilemap_index = shadow.tilemap_index + face_id;
  ShadowTileData tile = shadow_tile_load(tilemaps_tx, tile_co, 0, tilemap_index);

  float depth = 1.0;
  if ((tilemap_index <= shadow.tilemap_last) && (tile.is_allocated || tile.lod > 0)) {
    vec2 shadow_uv = shadow_page_uv_transform(tile.page, tile.lod, uv);
    depth = texture(atlas_tx, shadow_uv).r;
  }
  return depth;
}

float shadow_directional_depth_get(sampler2D atlas_tx,
                                   usampler2D tilemaps_tx,
                                   LightData light,
                                   ShadowData shadow,
                                   vec3 camera_P,
                                   vec3 lP,
                                   vec3 P)
{
  int clipmap_lod = shadow_directional_clipmap_level(shadow, distance(P, camera_P));
  int clipmap_lod_relative = clipmap_lod - shadow.clipmap_lod_min;
  int tilemap_index = clamp(
      shadow.tilemap_index + clipmap_lod_relative, shadow.tilemap_index, shadow.tilemap_last);
  /* Compute how many time we need to subdivide. */
  float clipmap_res_mul = float(1 << (shadow.clipmap_lod_max - clipmap_lod));
  /* Compute offset of the clipmap from the largest LOD. */
  vec2 clipmap_offset = vec2(abs(shadow.base_offset) >> clipmap_lod_relative) *
                        sign(shadow.base_offset);

  vec2 uv = (lP.xy * clipmap_res_mul - clipmap_offset) + float(SHADOW_TILEMAP_RES / 2);
  ivec2 tile_co = ivec2(floor(uv));
  ShadowTileData tile = shadow_tile_load(tilemaps_tx, tile_co, 0, tilemap_index);

  float depth = 1.0;
  if (tile.is_allocated) {
    vec2 shadow_uv = shadow_page_uv_transform(tile.page, 0, uv);
    depth = texture(atlas_tx, shadow_uv).r;
  }
  return depth;
}

/* Returns world distance delta from light between shading point and first occluder. */
float shadow_delta_get(sampler2D atlas_tx,
                       usampler2D tilemaps_tx,
                       LightData light,
                       ShadowData shadow,
                       vec3 lL,
                       float receiver_dist,
                       vec3 P)
{
  if (light.type == LIGHT_SUN) {
    /* [-SHADOW_TILEMAP_RES/2..SHADOW_TILEMAP_RES/2] range for highest LOD. */
    vec3 lP = transform_point(shadow.mat, P);
    float occluder_z = shadow_directional_depth_get(
        atlas_tx, tilemaps_tx, light, shadow, cameraPos, lP, P);
    /* Transform to world space distance. */
    return (lP.z - occluder_z) * abs(shadow.clip_far - shadow.clip_near);
  }
  else {
    float occluder_z = shadow_punctual_depth_get(atlas_tx, tilemaps_tx, light, shadow, lL);
    occluder_z = linear_depth(true, occluder_z, shadow.clip_far, shadow.clip_near);
    /* Take into account the cubemap projection. We want the radial distance. */
    float occluder_dist = receiver_dist * occluder_z / max_v3(abs(lL));
    return receiver_dist - occluder_dist;
  }
}

/** \} */
