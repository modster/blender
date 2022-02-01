
/**
 * Virtual shadowmapping: Free pages.
 *
 * Scan all tilemaps and add all free pages inside the free page heap.
 */

#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shadow_page_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shadow_tilemap_lib.glsl)

void main()
{
  ShadowTileMapData tilemap_data = tilemaps[gl_GlobalInvocationID.z];
  ivec2 tile_co = ivec2(gl_GlobalInvocationID.xy);
  int tilemap_idx = tilemap_data.index;

  int lod_max = tilemap_data.is_cubeface ? SHADOW_TILEMAP_LOD : 0;
  for (int lod = 0; lod <= lod_max; lod++) {
    uint lod_size = SHADOW_TILEMAP_RES >> lod;

    if (any(greaterThanEqual(tile_co, ivec2(lod_size)))) {
      continue;
    }

    ivec2 texel = shadow_tile_coord_in_atlas(tile_co, tilemap_idx, lod);
    ShadowTileData tile = shadow_tile_data_unpack(imageLoad(tilemaps_img, texel).x);

    if (tile.is_allocated) {
      if (tile.is_visible && tile.is_used && tile.is_cached) {
        /* Try to recover cached tiles. Update flag is kept untouched as content might be valid. */
        free_page_owners[tile.free_page_owner_index] = uint(-1);
        tile.is_cached = false;
        tile.free_page_owner_index = uint(-1);
        imageStore(tilemaps_img, texel, uvec4(shadow_tile_data_pack(tile)));
      }
      else if ((!tile.is_visible || !tile.is_used) && !tile.is_cached) {
        /* Push page to the free page heap. */
        int free_index = atomicAdd(infos.page_free_next, 1) + 1;
        if (free_index < SHADOW_MAX_PAGE) {
          free_page_owners[free_index] = packUvec2x16(uvec2(texel));
          tile.is_cached = true;
          tile.free_page_owner_index = uint(free_index);
          imageStore(tilemaps_img, texel, uvec4(shadow_tile_data_pack(tile)));
        }
        else {
          /* Well, this should never happen. This would mean some pages were marked
           * for deletion multiple times. */
        }
      }
    }
  }
}