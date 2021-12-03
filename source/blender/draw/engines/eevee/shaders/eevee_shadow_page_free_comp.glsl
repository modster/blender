
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

layout(r32ui) restrict uniform uimage2D tilemaps_img;

void main()
{
  ShadowTileMapData tilemap_data = tilemaps[gl_GlobalInvocationID.z];
  ivec2 tile_co = ivec2(gl_GlobalInvocationID.xy);
  int tilemap_idx = tilemap_data.index;

  /* Bitmap of intersection tests. Use one uint per lod tile. */
  /* TODO(fclem): We use only 4bits out of the 32bits per uint. Pack this better. */
  shared uint intersect_map[SHADOW_TILEMAP_RES * SHADOW_TILEMAP_RES / 4];

  int lod_max = tilemap_data.is_cubeface ? SHADOW_TILEMAP_LOD : 0;
  for (int lod = 0; lod <= lod_max; lod++) {
    ivec2 tile_co = ivec2(gl_GlobalInvocationID.xy) >> lod;
    uint stride = 1u << lod;
    uint lod_size = SHADOW_TILEMAP_RES >> lod;
    /* We load the same data for each thread covering the same LOD tile, but we avoid
     * freeing the same tile twice. This is because we need uniform control flow for the
     * barriers to be valid. */
    bool valid_thread = (gl_GlobalInvocationID.xy % stride) == uvec2(0);

    ivec2 texel = shadow_tile_coord_in_atlas(tile_co, tilemap_idx, lod);
    ShadowTileData tile = shadow_tile_data_unpack(imageLoad(tilemaps_img, texel).x);

    bool tile_lod_is_covered = false;

    if (lod > 0) {
      /* We mask tiles from lower LOD that are completely covered by the lods above it. */
      if ((intersect_map[tile_co.y * lod_size + tile_co.x] & 0xFu) == 0xFu) {
        tile.is_used = false;
        tile_lod_is_covered = true;
      }
    }
    barrier();

    if (valid_thread && (!tile.is_visible || !tile.is_used) && tile.is_allocated) {
      /* Push page to the free page heap. */
      uint page_index = shadow_page_to_index(tile.page);
#ifdef SHADOW_DEBUG_PAGE_ALLOCATION_ENABLED
      pages[page_index] = SHADOW_PAGE_NO_DATA;
#endif

      int free_index = atomicAdd(SHADOW_PAGE_HEAP_LAST_USED(pages_free), 1) + 1;
      if (free_index < SHADOW_MAX_PAGE - 1) {
        pages_free[free_index] = int(page_index);
      }
      else {
        /* Well, this should never happen. This would mean some pages were marked
         * for deletion multiple times. */
        SHADOW_PAGE_HEAP_LAST_USED(pages_free) = SHADOW_MAX_PAGE - 2;
      }

      tile.is_allocated = false;
#ifdef SHADOW_DEBUG_TILE_ALLOCATION_ENABLED
      tile.page = uvec2(0);
      tile.is_visible = false;
      tile.do_update = false;
      tile.is_used = false;
#endif
    }

    imageStore(tilemaps_img, texel, uvec4(shadow_tile_data_pack(tile)));

    if (valid_thread) {
      uint tile_shift = (uint(tile_co.x) & 0x1u) + (uint(tile_co.y) & 0x1u) * 2u;
      tile_co >>= 1;
      lod_size >>= 1;

      if ((tile.is_used && tile.is_visible) || tile_lod_is_covered) {
        atomicOr(intersect_map[tile_co.y * lod_size + tile_co.x], 1u << tile_shift);
      }
      else {
        atomicAnd(intersect_map[tile_co.y * lod_size + tile_co.x], ~(1u << tile_shift));
      }
    }
    barrier();
  }
}