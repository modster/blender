
/**
 * Virtual shadowmapping: Free pages.
 *
 * Scan all tilemaps and add all free pages inside the free page heap.
 */

#pragma BLENDER_REQUIRE(eevee_shadow_page_ops_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shadow_tilemap_lib.glsl)

void main()
{
  ShadowTileMapData tilemap_data = tilemaps_buf[gl_GlobalInvocationID.z];
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

    bool tile_is_needed = tile.is_visible && tile.is_used;

    if (!tile_is_needed) {
      if (tile.do_update && tile.is_allocated) {
        /* Directly free. The tile would need to be updated anyway. */
        shadow_page_free_buf_append(tile);
        imageStore(tilemaps_img, texel, uvec4(shadow_tile_data_pack(tile)));
      }
      else if (tile.do_update && tile.is_cached) {
        /* Was previously pushed to cache but has been tagged for update. No need to keep it. */
        shadow_page_cached_buf_remove(tile);
        shadow_page_free_buf_append(tile);
        imageStore(tilemaps_img, texel, uvec4(shadow_tile_data_pack(tile)));
      }
      else if (!tile.is_cached && tile.is_allocated) {
        /* The page is just being unused and hasn't been pushed to the cache yet. */
        shadow_page_cached_buf_append(tile, texel);
        imageStore(tilemaps_img, texel, uvec4(shadow_tile_data_pack(tile)));
      }
    }
    else {
      if (tile.is_cached) {
        /* Reuse cached page, even if the tile has been tagged for update. */
        shadow_page_cached_buf_remove(tile);
        imageStore(tilemaps_img, texel, uvec4(shadow_tile_data_pack(tile)));
      }
      else {
        /* Count the number of needed pages for allocation. The defrag step will free some cached
         * pages if there is not enough free pages. */
        atomicAdd(pages_infos_buf.page_alloc_count, 1);
      }
    }
  }
}