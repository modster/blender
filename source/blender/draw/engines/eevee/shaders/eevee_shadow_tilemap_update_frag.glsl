
/**
 * Virtual shadowmapping: Shadow casters update phase for tilemaps.
 * We render points sprites to tag every tiles where a shadow caster was updated.
 */

#pragma BLENDER_REQUIRE(eevee_shadow_tilemap_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shadow_aabb_interface_lib.glsl)

layout(r32ui) restrict uniform uimage2D tilemaps_img;

void main()
{
  ivec2 tile_co = ivec2(gl_FragCoord.xy);
  ShadowTileData tile_data = shadow_tile_load(tilemaps_img, tile_co, interp.tilemap_index);
  /* NOTE: is_used imply is_visible. */
  if (tile_data.is_used && !tile_data.do_update && aabb_raster(gl_FragCoord.xy)) {
    ivec2 tile_co = ivec2(gl_FragCoord.xy);
    shadow_tile_set_flag(tilemaps_img, tile_co, interp.tilemap_index, SHADOW_TILE_DO_UPDATE);
  }
}
