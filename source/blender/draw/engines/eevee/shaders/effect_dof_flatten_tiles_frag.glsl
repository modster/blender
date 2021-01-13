
/**
 * Tile flatten pass: Takes the halfres CoC buffer and converts it to 8x8 tiles.
 *
 * Output min and max values for each tile and for both foreground & background.
 * Also outputs min intersectable CoC for the background, which is the minimum CoC
 * that comes from the background pixels.
 **/

#pragma BLENDER_REQUIRE(effect_dof_lib.glsl)

/* Half resolution. */
uniform sampler2D halfResCocBuffer;

/* 1/8th of halfResCocBuffer resolution. So 1/16th of fullres. */
layout(location = 0) out vec2 outFgCoc; /* Min, Max */
layout(location = 1) out vec3 outBgCoc; /* Min, Max, MinIntersectable */

void main()
{
  vec2 halfres_texel_size = 1.0 / vec2(textureSize(halfResCocBuffer, 0).xy);

  CocTile tile = dof_coc_tile_init();

  for (int x = 0; x < 8; x++) {
    /* OPTI: Could be done in separate passes. */
    for (int y = 0; y < 8; y++) {
      vec2 sample_uv = (floor(gl_FragCoord.xy) * 8.0 + vec2(x, y) + 0.5) * halfres_texel_size;
      float sample_coc = textureLod(halfResCocBuffer, sample_uv, 0.0).r;

      float fg_coc = min(sample_coc, 0.0);
      tile.fg_min_coc = min(tile.fg_min_coc, fg_coc);
      tile.fg_max_coc = max(tile.fg_max_coc, fg_coc);

      float bg_coc = max(sample_coc, 0.0);
      tile.bg_min_coc = min(tile.bg_min_coc, bg_coc);
      tile.bg_max_coc = max(tile.bg_max_coc, bg_coc);

      if (sample_coc > 0.0) {
        tile.bg_min_intersectable_coc = min(tile.bg_min_intersectable_coc, bg_coc);
      }
    }
  }

  outFgCoc = vec2(tile.fg_min_coc, tile.fg_max_coc);
  outBgCoc = vec3(tile.bg_min_coc, tile.bg_max_coc, tile.bg_min_intersectable_coc);
}
