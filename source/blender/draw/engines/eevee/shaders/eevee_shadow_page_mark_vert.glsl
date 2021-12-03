
/**
 * Virtual shadowmapping: Page marking / preparation
 *
 * This renders a series of quad to needed pages render locations.
 * This is in order to clear the depth to 1.0 only where it is needed
 * to occlude any potentially costly fragment shader invocation.
 */

#pragma BLENDER_REQUIRE(eevee_shadow_tilemap_lib.glsl)

uniform usampler2D tilemaps_tx;

uniform int tilemap_index;
uniform int tilemap_lod;

void main()
{
  int tile_index = gl_VertexID / 6;
  ivec2 tile_co = ivec2(tile_index % SHADOW_TILEMAP_RES, tile_index / SHADOW_TILEMAP_RES);
  tile_co >>= tilemap_lod;

  ShadowTileData tile = shadow_tile_load(tilemaps_tx, tile_co, tilemap_lod, tilemap_index);

  if (!tile.is_visible || !tile.is_used || !tile.do_update) {
    /* Don't draw anything as we already cleared the render target for these areas. */
    gl_Position = vec4(0.0);
    return;
  }

  int v = gl_VertexID % 3;
  /* Triangle in lower left corner in [-1..1] square. */
  vec2 pos = -1.0 + vec2((v & 1) << 1, (v & 2) << 0);
  /* NOTE: this only renders if backface cull is off. */
  pos = ((gl_VertexID % 6) > 2) ? -pos : pos;

  pos = ((pos * 0.5 + 0.5) + vec2(tile_co)) / float(SHADOW_TILEMAP_RES >> tilemap_lod);

  gl_Position = vec4(pos * 2.0 - 1.0, 1.0, 1.0);
}