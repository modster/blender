
/**
 * Virtual shadowmapping: Shadow receiver update phase for tilemaps.
 * We render points sprites to tag every tiles where shadow receivers are.
 * This is designed to support alpha blended geometry that does not necessarily write depth.
 */

#pragma BLENDER_REQUIRE(eevee_shadow_tilemap_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shadow_aabb_interface_lib.glsl)

layout(r32ui) restrict uniform uimage2D tilemaps_img;

void main()
{
  ivec2 tile_co = ivec2(gl_FragCoord.xy);
  ShadowTileData tile_data = shadow_tile_load(tilemaps_img, tile_co, interp.tilemap_index);

  if (!tile_data.is_used && tile_data.is_visible && aabb_raster(gl_FragCoord.xy)) {
    shadow_tile_set_flag(tilemaps_img, tile_co, interp.tilemap_index, SHADOW_TILE_IS_USED);
  }
}
