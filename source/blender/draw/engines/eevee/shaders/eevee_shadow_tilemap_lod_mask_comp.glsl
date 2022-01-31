
/**
 * Virtual shadowmapping: LOD mask.
 *
 * Discard pages that are redundant in the mipmap chain.
 * We mask tiles that are completely covered by higher LOD tiles.
 */

#pragma BLENDER_REQUIRE(eevee_shadow_tilemap_lib.glsl)

layout(local_size_x = SHADOW_TILEMAP_RES, local_size_y = SHADOW_TILEMAP_RES) in;

layout(std430, binding = 0) restrict readonly buffer tilemaps_buf
{
  ShadowTileMapData tilemaps[];
};

layout(r32ui) restrict uniform uimage2D tilemaps_img;

void main()
{
  ShadowTileMapData tilemap_data = tilemaps[gl_GlobalInvocationID.z];
  int tilemap_idx = tilemap_data.index;
  int lod_max = tilemap_data.is_cubeface ? SHADOW_TILEMAP_LOD : 0;

  /* Bitmap of usage tests. Use one uint per tile. One bit per lod level. */
  shared uint lod_map[SHADOW_TILEMAP_RES * SHADOW_TILEMAP_RES];

  /* For now there is nothing to do for directional shadows. */
  if (tilemap_data.is_cubeface) {
    ivec2 tile_co = ivec2(gl_GlobalInvocationID.xy);

    lod_map[SHADOW_TILEMAP_RES * tile_co.y + tile_co.x] = 0u;

    int map_index = tile_co.y * SHADOW_TILEMAP_RES + tile_co.x;
    for (int lod = 0; lod <= lod_max; lod++) {
      ivec2 tile_co_lod = ivec2(gl_GlobalInvocationID.xy) >> lod;
      ShadowTileData tile = shadow_tile_load(tilemaps_img, tile_co_lod, lod, tilemap_idx);

      if (tile.is_used && tile.is_visible) {
        lod_map[map_index] |= 1u << uint(lod);
      }
    }

    barrier();

    /* We mask tiles from lower LOD that are completely covered by the lods above it. */
    for (int lod = 1; lod <= SHADOW_TILEMAP_LOD; lod++) {
      uint stride = 1u << uint(lod);
      if ((gl_GlobalInvocationID.xy % stride) != uvec2(0)) {
        continue;
      }
      uint lod_mask = ~(~0x0u << uint(lod));
      bool tiles_covered = true;
      for (int x = 0; x < stride && tiles_covered; x++) {
        for (int y = 0; y < stride && tiles_covered; y++) {
          ivec2 tile_co = ivec2(gl_GlobalInvocationID.xy) + ivec2(x, y);
          uint lod_bits = lod_map[tile_co.y * SHADOW_TILEMAP_RES + tile_co.x];
          if ((lod_bits & lod_mask) == 0u) {
            tiles_covered = false;
          }
        }
      }
      if (tiles_covered) {
        ivec2 tile_co = ivec2(gl_GlobalInvocationID.xy >> uint(lod));
        shadow_tile_unset_flag(tilemaps_img, tile_co, lod, tilemap_idx, SHADOW_TILE_IS_USED);
      }
    }
  }
}