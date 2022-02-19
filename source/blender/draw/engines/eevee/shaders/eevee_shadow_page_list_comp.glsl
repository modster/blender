
/**
 * Virtual shadowmapping: Render Page list creation.
 * For the given tilemap, scan through all tiles and create a buffer with only the updated tiles.
 */

#pragma BLENDER_REQUIRE(eevee_shadow_tilemap_lib.glsl)

void main()
{
  if (gl_GlobalInvocationID == uvec3(0)) {
    pages_infos_buf.page_rendered = 0;
  }
  barrier();

  ivec2 tile_co = ivec2(gl_LocalInvocationID.xy);

  ShadowTileData tile = shadow_tile_load_img(tilemaps_img, tile_co, tilemap_lod, tilemap_index);

  if (tile.do_update && tile.is_allocated) {
    uint index = atomicAdd(pages_infos_buf.page_rendered, 1);
    pages_list_buf[index] = packUvec4x8(uvec4(tile_co, tile.page));
    shadow_tile_unset_flag(
        tilemaps_img, tile_co, tilemap_lod, tilemap_index, SHADOW_TILE_DO_UPDATE);
  }
}