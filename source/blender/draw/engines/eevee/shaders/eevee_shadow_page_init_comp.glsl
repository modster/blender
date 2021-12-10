
/**
 * Virtual shadowmapping: Init page buffer.
 *
 * All pages are always owned by tiles. This step init all owners.
 * This avoid mapping the buffer to host memory.
 */

#pragma BLENDER_REQUIRE(eevee_shadow_page_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shadow_tilemap_lib.glsl)

layout(local_size_x = SHADOW_PAGE_PER_ROW) in;

layout(std430, binding = 1) restrict writeonly buffer pages_free_buf
{
  uint free_page_owners[];
};

layout(std430, binding = 3) restrict writeonly buffer pages_infos_buf
{
  ShadowPagesInfoData infos;
};

layout(r32ui) restrict uniform uimage2D tilemaps_img;

void main()
{
  if (gl_GlobalInvocationID == uvec3(0)) {
    infos.page_free_next = SHADOW_MAX_PAGE - 1;
    infos.page_free_next_prev = 0;
    infos.page_updated_count = 0;
  }

  uint page_index = gl_GlobalInvocationID.x;

  ivec2 texel = ivec2(page_index % (SHADOW_TILEMAP_PER_ROW * SHADOW_TILEMAP_RES),
                      page_index / (SHADOW_TILEMAP_PER_ROW * SHADOW_TILEMAP_RES));
  free_page_owners[page_index] = packUvec2x16(uvec2(texel));

  /* Start with a blank tile. */
  ShadowTileData tile = shadow_tile_data_unpack(0u);
  tile.page = uvec2(page_index % uint(SHADOW_PAGE_PER_ROW),
                    page_index / uint(SHADOW_PAGE_PER_ROW));
  tile.free_page_owner_index = page_index;
  tile.is_allocated = true;
  tile.is_cached = true;
  tile.do_update = true;
  imageStore(tilemaps_img, texel, uvec4(shadow_tile_data_pack(tile)));
}