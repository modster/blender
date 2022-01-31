
/**
 * Virtual shadowmapping: Setup phase for tilemaps.
 * During this phase we clear the visibility, usage and request bits.
 * This is also where we shifts the whole tilemap for directional shadow clipmaps
 */

#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)
#pragma BLENDER_REQUIRE(eevee_shadow_page_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shadow_tilemap_lib.glsl)

layout(local_size_x = SHADOW_TILEMAP_RES, local_size_y = SHADOW_TILEMAP_RES) in;

layout(std430, binding = 0) readonly buffer tilemaps_buf
{
  ShadowTileMapData tilemaps[];
};

layout(std430, binding = 1) restrict buffer pages_free_buf
{
  uint free_page_owners[];
};

layout(std430, binding = 3) restrict buffer pages_infos_buf
{
  ShadowPagesInfoData infos;
};

layout(r32ui) restrict uniform uimage2D tilemaps_img;

uniform bool do_tilemap_setup;

void main()
{
  ShadowTileMapData tilemap = tilemaps[gl_GlobalInvocationID.z];
  ivec2 grid_shift = (do_tilemap_setup) ? tilemap.grid_shift : ivec2(0);
  ivec2 tile_co = ivec2(gl_GlobalInvocationID.xy);
  ivec2 tile_shifted = tile_co + grid_shift;
  /* Still load a valid tile after the shifting in order to not loose any page reference.
   * This way the tile can even be reused if it is needed. Also avoid negative modulo. */
  ivec2 tile_wrapped = (tile_shifted + SHADOW_TILEMAP_RES) % SHADOW_TILEMAP_RES;

  ivec2 texel_out = shadow_tile_coord_in_atlas(tile_co, tilemap.index, 0);
  ivec2 texel_in = shadow_tile_coord_in_atlas(tile_wrapped, tilemap.index, 0);

  ShadowTileData tile_data = shadow_tile_data_unpack(imageLoad(tilemaps_img, texel_in).x);
  tile_data.is_visible = false;
  tile_data.is_used = false;
  tile_data.lod = 0;

  if (!in_range_inclusive(tile_shifted, ivec2(0), ivec2(SHADOW_TILEMAP_RES - 1))) {
    /* This tile was shifted in. */
    tile_data.do_update = true;
  }

  if (grid_shift != ivec2(0) && tile_data.is_cached) {
    /* Update page location after shift. */
    free_page_owners[tile_data.free_page_owner_index] = packUvec2x16(uvec2(texel_out));
  }

  imageStore(tilemaps_img, texel_out, uvec4(shadow_tile_data_pack(tile_data)));

  if (tilemap.is_cubeface) {
    /* Cubemap shift update is always all or nothing. */
    bool do_update = (grid_shift.x != 0);

    /* Number of lod0 tiles covered by the current lod level (in one dimension). */
    uint lod_stride = 1u << 1u;
    uint lod_size = uint(SHADOW_TILEMAP_RES) >> 1u;
    for (int lod = 1; lod <= SHADOW_TILEMAP_LOD; lod++, lod_size >>= 1u, lod_stride <<= 1u) {
      if (all(lessThan(tile_co, ivec2(lod_size)))) {
        ivec2 texel = shadow_tile_coord_in_atlas(tile_co, tilemap.index, lod);

        ShadowTileData tile_data = shadow_tile_data_unpack(imageLoad(tilemaps_img, texel).x);
        tile_data.is_visible = false;
        tile_data.is_used = false;
        tile_data.do_update = do_update;
        tile_data.lod = 0;
        imageStore(tilemaps_img, texel, uvec4(shadow_tile_data_pack(tile_data)));
      }
    }
  }

  if (gl_GlobalInvocationID == uvec3(0)) {
    infos.page_free_next = max(-1, infos.page_free_next);
    infos.page_free_next_prev = infos.page_free_next;
    infos.page_updated_count = 0;
  }
}