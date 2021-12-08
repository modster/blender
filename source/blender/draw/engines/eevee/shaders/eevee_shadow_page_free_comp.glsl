
/**
 * Virtual shadowmapping: Free pages.
 *
 * Scan all tilemaps and add all free pages inside the free page heap.
 */

#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shadow_page_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shadow_tilemap_lib.glsl)

layout(local_size_x = SHADOW_TILEMAP_RES, local_size_y = SHADOW_TILEMAP_RES) in;

layout(std430, binding = 0) restrict readonly buffer tilemaps_buf
{
  ShadowTileMapData tilemaps[];
};

layout(std430, binding = 1) restrict buffer pages_free_buf
{
  int pages_free[];
};

layout(std430, binding = 2) restrict writeonly buffer pages_buf
{
  ShadowPagePacked pages[];
};

layout(std430, binding = 3) restrict buffer pages_infos_buf
{
  ShadowPagesInfoData infos;
};

layout(r32ui) restrict uniform uimage2D tilemaps_img;

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

    if (tile.is_allocated && (!tile.is_visible || !tile.is_used)) {
      /* Push page to the free page heap. */
      uint page_index = shadow_page_to_index(tile.page);
#ifdef SHADOW_DEBUG_PAGE_ALLOCATION_ENABLED
      pages[page_index] = SHADOW_PAGE_NO_DATA;
#endif

      int free_index = atomicAdd(infos.page_free_next, 1) + 1;
      if (free_index < SHADOW_MAX_PAGE) {
        pages_free[free_index] = int(page_index);
      }
      else {
        /* Well, this should never happen. This would mean some pages were marked
         * for deletion multiple times. */
        infos.page_free_next = SHADOW_MAX_PAGE - 1;
      }

      tile.is_allocated = false;
#ifdef SHADOW_DEBUG_TILE_ALLOCATION_ENABLED
      tile.page = uvec2(0);
      tile.is_visible = false;
      tile.do_update = false;
      tile.is_used = false;
#endif
      imageStore(tilemaps_img, texel, uvec4(shadow_tile_data_pack(tile)));
    }
  }
}