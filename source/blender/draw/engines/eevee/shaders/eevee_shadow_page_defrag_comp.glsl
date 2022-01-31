
/**
 * Virtual shadowmapping: Defrag.
 *
 * Defragment the free page owner heap making one continuous array.
 */

#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shadow_page_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shadow_tilemap_lib.glsl)

layout(local_size_x = 1) in;

layout(std430, binding = 1) restrict buffer pages_free_buf
{
  uint free_page_owners[];
};

layout(std430, binding = 3) restrict buffer pages_infos_buf
{
  ShadowPagesInfoData infos;
};

layout(r32ui) restrict uniform uimage2D tilemaps_img;

void find_last_valid(inout uint last_valid)
{
  for (uint i = last_valid; i > 0u; i--) {
    if (free_page_owners[i] != uint(-1)) {
      last_valid = i;
      break;
    }
  }
}

void main()
{
  uint last_valid = uint(infos.page_free_next);

  find_last_valid(last_valid);

  for (uint i = 0u; i < last_valid; i++) {
    if (free_page_owners[i] == uint(-1)) {
      free_page_owners[i] = free_page_owners[last_valid];

      /* Update corresponding reference in tile. */
      ivec2 texel = ivec2(unpackUvec2x16(free_page_owners[last_valid]));
      ShadowTileData tile = shadow_tile_data_unpack(imageLoad(tilemaps_img, texel).x);
      tile.free_page_owner_index = i;
      imageStore(tilemaps_img, texel, uvec4(shadow_tile_data_pack(tile)));

      free_page_owners[last_valid] = uint(-1);
      find_last_valid(last_valid);
    }
  }

  infos.page_free_next = int(last_valid);
}