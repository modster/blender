
/**
 * Virtual shadowmapping: Defrag.
 *
 * Defragment the free page owner heap making one continuous array.
 */

#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shadow_page_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shadow_tilemap_lib.glsl)

void find_last_valid(inout uint last_valid)
{
  for (uint i = last_valid; i > 0u; i--) {
    if (pages_free_buf[i] != uint(-1)) {
      last_valid = i;
      break;
    }
  }
}

void main()
{
  uint last_valid = uint(pages_infos_buf.page_free_next);

  find_last_valid(last_valid);

  for (uint i = 0u; i < last_valid; i++) {
    if (pages_free_buf[i] == uint(-1)) {
      pages_free_buf[i] = pages_free_buf[last_valid];

      /* Update corresponding reference in tile. */
      ivec2 texel = ivec2(unpackUvec2x16(pages_free_buf[last_valid]));
      ShadowTileData tile = shadow_tile_data_unpack(imageLoad(tilemaps_img, texel).x);
      tile.free_page_owner_index = i;
      imageStore(tilemaps_img, texel, uvec4(shadow_tile_data_pack(tile)));

      pages_free_buf[last_valid] = uint(-1);
      find_last_valid(last_valid);
    }
  }

  pages_infos_buf.page_free_next = int(last_valid);
}