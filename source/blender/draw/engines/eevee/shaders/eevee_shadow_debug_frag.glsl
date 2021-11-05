
/**
 * Debug drawing for virtual shadowmaps.
 * See eShadowDebug for more information.
 */

#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shadow_tilemap_lib.glsl)

/** Control the scaling of the tilemap splat. */
const float pixel_scale = 5.0;

layout(std140) uniform debug_block
{
  ShadowDebugData debug;
};

uniform usampler2D tilemaps_tx;
uniform sampler2D depth_tx;

layout(location = 0, index = 0) out vec4 out_color_add;
layout(location = 0, index = 1) out vec4 out_color_mul;

void debug_tilemap()
{
  ivec2 tile = ivec2(gl_FragCoord.xy / pixel_scale);
  int tilemap_lod = tile.x / (SHADOW_TILEMAP_RES + 2);
  int tilemap_index = tile.y / (SHADOW_TILEMAP_RES + 2);
  tile = (tile % (SHADOW_TILEMAP_RES + 2)) - 1;
  tilemap_index += debug.shadow.tilemap_index;

  if ((tilemap_index >= debug.shadow.tilemap_index) &&
      (tilemap_index <= debug.shadow.tilemap_last) && (tilemap_lod == 0) &&
      in_range_inclusive(tile, ivec2(0), ivec2(SHADOW_TILEMAP_RES - 1))) {
    ShadowTileData tile = shadow_tile_load(tilemaps_tx, tile, tilemap_index);

    if (tile.do_update && tile.is_used && tile.is_visible) {
      out_color_add = vec4(1, 0, 0, 0);
    }
    else if (tile.is_used && tile.is_visible) {
      out_color_add = vec4(0, 1, 0, 0);
    }
    else if (tile.is_visible) {
      out_color_add = vec4(0, 0.2, 0.8, 0);
    }
    gl_FragDepth = 0.0;
    out_color_mul = vec4(0);
  }
}

void main()
{
  /* Default to no output. */
  gl_FragDepth = 1.0;
  out_color_add = vec4(0.0);
  out_color_mul = vec4(1.0);

  debug_tilemap();
}