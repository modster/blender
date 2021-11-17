
/**
 * Virtual shadowmapping: Setup phase for tilemaps.
 * During this phase we clear the visibility, usage and request bits.
 * This is also where we shifts the whole tilemap for directional shadow clipmaps
 */

#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)
#pragma BLENDER_REQUIRE(eevee_shadow_tilemap_lib.glsl)

layout(local_size_x = SHADOW_TILEMAP_RES, local_size_y = SHADOW_TILEMAP_RES) in;

layout(std430, binding = 0) readonly buffer tilemaps_buf
{
  ShadowTileMapData tilemaps[];
};

layout(r32ui) restrict uniform uimage2D tilemaps_img;

void main()
{
  ShadowTileMapData tilemap = tilemaps[gl_GlobalInvocationID.z];

  ivec2 tile_co = ivec2(gl_GlobalInvocationID.xy);
  ivec2 tile_shifted = tile_co + tilemap.grid_shift;
  /* Still load a valid tile after the shifting in order to not loose any page reference.
   * This way the tile can even be reused if it is needed. Also avoid negative modulo. */
  ivec2 tile_wrapped = (tile_shifted + SHADOW_TILEMAP_RES) % SHADOW_TILEMAP_RES;

  ShadowTileData tile_data = shadow_tile_load(tilemaps_img, tile_wrapped, tilemap.index);
  /* Reset all flags but keep the allocated page. */
  tile_data.is_visible = false;
  tile_data.is_used = false;
  tile_data.do_update = false;
  tile_data.lod_tilemap_offset = 0;
#ifdef SHADOW_NO_CACHING
  tile_data.page = uvec2(0);
  tile_data.is_allocated = false;
#endif

  if (!in_range_inclusive(tile_shifted, ivec2(0), ivec2(SHADOW_TILEMAP_RES - 1))) {
    /* This tile was shifted in. */
    tile_data.do_update = true;
  }

  shadow_tile_store(tilemaps_img, tile_co, tilemap.index, tile_data);

  if (tilemap.is_cubeface) {
    /* TODO(fclem) Need array of image. Do recursive downsample using groupshared. */
  }
}