
/**
 * Virtual shadowmapping: Tile copy.
 *
 * This pass copies the pages rendered in the render target to the page atlas.
 * This might not be the fastest way to blit shadow regions but at least it is fully GPU driven.
 */

/* TODO(fclem): The goal would be to render on the atlas texture and only move pages if
 * they overlap with the rendering. */

#pragma BLENDER_REQUIRE(eevee_shadow_tilemap_lib.glsl)

bool do_corner_pattern(vec2 uv, float corner_ratio)
{
  uv = uv * 2.0 - 1.0;
  return (any(greaterThan(abs(uv), vec2(1.0 - corner_ratio))) &&
          all(greaterThan(abs(uv), vec2(1.0 - corner_ratio * 2.0))));
}

void main()
{
  int page_size = textureSize(render_tx, 0).x / SHADOW_TILEMAP_RES;

  for (int p = 0; p < pages_infos_buf.page_rendered; p++) {
    uvec4 render_tile = unpackUvec4x8(pages_list_buf[p]);

    ivec2 tile_co = ivec2(render_tile.xy);
    ivec2 page_co = ivec2(render_tile.zw);

    /* We dispatch enough group to cover one page. */
    ivec2 page_texel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 in_texel = page_texel + tile_co * page_size;
    ivec2 out_texel = page_texel + page_co * page_size;

    float depth = texelFetch(render_tx, in_texel, 0).r;

    /* Debugging. */
    uvec2 page_size = gl_NumWorkGroups.xy * gl_WorkGroupSize.xy;
    vec2 uv = vec2(gl_GlobalInvocationID.xy) / vec2(page_size);
    if (do_corner_pattern(uv, 0.05)) {
      depth = 0.0;
    }
    else if (do_corner_pattern(uv, 0.08)) {
      depth = 1.0;
    }

    // if (do_char()) {

    // }

    imageStore(out_atlas_img, out_texel, vec4(depth));
  }
}