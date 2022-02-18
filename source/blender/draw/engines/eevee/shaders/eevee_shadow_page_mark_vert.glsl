
/**
 * Virtual shadowmapping: Page marking / preparation
 *
 * This renders a series of quad to needed pages render locations.
 * This is in order to clear the depth to 1.0 only where it is needed
 * to occlude any potentially costly fragment shader invocation.
 */

#pragma BLENDER_REQUIRE(eevee_shadow_tilemap_lib.glsl)

void main()
{
  int tile_index = gl_VertexID / 6;

  if (tile_index >= pages_infos_buf.page_rendered) {
    gl_Position = vec4(0.0);
    return;
  }

  uvec2 render_co = unpackUvec4x8(pages_list_buf[tile_index]).xy;

  int v = gl_VertexID % 3;
  /* Triangle in lower left corner in [-1..1] square. */
  vec2 pos = -1.0 + vec2((v & 1) << 1, (v & 2) << 0);
  /* NOTE: this only renders if backface cull is off. */
  pos = ((gl_VertexID % 6) > 2) ? -pos : pos;

  pos = ((pos * 0.5 + 0.5) + vec2(render_co)) / float(SHADOW_TILEMAP_RES >> tilemap_lod);

  gl_Position = vec4(pos * 2.0 - 1.0, 1.0, 1.0);
}