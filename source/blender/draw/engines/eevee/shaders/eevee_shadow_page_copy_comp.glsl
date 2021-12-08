
/**
 * Virtual shadowmapping: Tile copy.
 *
 * This pass copies the pages rendered in the render target to the page atlas.
 * This might not be the fastest way to blit shadow regions but at least it is fully GPU driven.
 */

/* TODO(fclem): The goal would be to render on the atlas texture and only move pages if
 * they overlap with the rendering. */

#pragma BLENDER_REQUIRE(eevee_shadow_tilemap_lib.glsl)

layout(local_size_x = SHADOW_PAGE_COPY_GROUP_SIZE, local_size_y = SHADOW_PAGE_COPY_GROUP_SIZE) in;

uniform usampler2D tilemaps_tx;
uniform sampler2D render_tx;

/* TODO(fclem): 16bit format. */
layout(r32f) writeonly restrict uniform image2D out_atlas_img;

uniform int tilemap_index;
uniform int tilemap_lod;

void main()
{
  int page_size = textureSize(render_tx, 0).x / SHADOW_TILEMAP_RES;
  int lod_size = SHADOW_TILEMAP_RES >> tilemap_lod;
  /* TODO(fclem) Experiment with biggest dispatch instead of iterating. Or a list. */
  for (int y = 0; y < lod_size; y++) {
    for (int x = 0; x < lod_size; x++) {
      ivec2 tile_co = ivec2(x, y);
      ShadowTileData tile = shadow_tile_load(tilemaps_tx, tile_co, tilemap_lod, tilemap_index);
      if (tile.do_update && tile.is_used && tile.is_visible && tile.is_allocated) {
        /* We dispatch enough group to cover one page. */
        ivec2 page_texel = ivec2(gl_GlobalInvocationID.xy);
        ivec2 in_texel = page_texel + tile_co * page_size;
        ivec2 out_texel = page_texel + ivec2(tile.page) * page_size;

        float depth = texelFetch(render_tx, in_texel, 0).r;
        imageStore(out_atlas_img, out_texel, vec4(depth));
      }
    }
  }
}