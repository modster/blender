
/* Recombine Pass
 * TODO decription
 */

#pragma BLENDER_REQUIRE(common_utiltex_lib.glsl)
#pragma BLENDER_REQUIRE(effect_dof_lib.glsl)

uniform sampler2D fullResColorBuffer;
uniform sampler2D fullResDepthBuffer;
uniform sampler2D bgColorBuffer;
uniform sampler2D bgWeightBuffer;
uniform sampler2D bgTileBuffer;
uniform sampler2D fgColorBuffer;
uniform sampler2D fgWeightBuffer;
uniform sampler2D fgTileBuffer;

in vec4 uvcoordsvar;

out vec4 fragColor;

void dof_slight_focus_gather(float radius, out vec4 out_color, out float out_weight)
{
  DofGatherData fg_accum = GATHER_DATA_INIT;
  DofGatherData bg_accum = GATHER_DATA_INIT;

  const int i_radius = 3;  // int(floor(radius)); /* TODO */
  ivec2 texel = ivec2(gl_FragCoord.xy);

  bool first_ring = true;

  for (int ring = 0; ring < i_radius; ring++) {
    DofGatherData fg_ring = GATHER_DATA_INIT;
    DofGatherData bg_ring = GATHER_DATA_INIT;

    int ring_distance = ring + 1;
    int ring_sample_count = 4 * ring_distance;
    for (int sample_id = 0; sample_id < ring_sample_count; sample_id++) {
      ivec2 offset = dof_square_ring_sample_offset(ring_distance, sample_id);
      float dist = length(vec2(offset));

      DofGatherData pair_data[2];
      for (int i = 0; i < 2; i++) {
        ivec2 sample_texel = texel + ((i == 0) ? offset : -offset);
        float depth = texelFetch(fullResDepthBuffer, sample_texel, 0).r;
        pair_data[i].color = texelFetch(fullResColorBuffer, sample_texel, 0);
        pair_data[i].coc = dof_coc_from_zdepth(depth);
        pair_data[i].dist = dist;
      }

      dof_gather_accumulate_sample_pair(
          pair_data, dist, first_ring, false, false, bg_ring, bg_accum);
      dof_gather_accumulate_sample_pair(
          pair_data, dist, first_ring, false, true, fg_ring, fg_accum);
    }

    dof_gather_accumulate_sample_ring(
        bg_ring, ring_sample_count, first_ring, false, false, bg_accum);
    dof_gather_accumulate_sample_ring(
        fg_ring, ring_sample_count, first_ring, false, true, fg_accum);

    first_ring = false;
  }

  /* Center sample. */
  float depth = texelFetch(fullResDepthBuffer, texel, 0).r;
  DofGatherData center_data;
  center_data.color = texelFetch(fullResColorBuffer, texel, 0);
  center_data.coc = dof_coc_from_zdepth(depth);
  center_data.dist = 0.0;

  dof_gather_accumulate_center_sample(center_data, false, true, fg_accum);
  dof_gather_accumulate_center_sample(center_data, false, false, bg_accum);

  vec4 bg_col, fg_col;
  float bg_weight, fg_weight;

  dof_gather_accumulate_resolve(i_radius, bg_accum, bg_col, bg_weight);
  dof_gather_accumulate_resolve(i_radius, fg_accum, fg_col, fg_weight);

  /* Fix weighting issues on perfectly focus > slight focus transitionning areas. */
  if (abs(center_data.coc) < 0.5) {
    bg_col = center_data.color;
    bg_weight = 1.0;
  }

  /* Alpha Over */
  float alpha = 1.0 - fg_weight;
  out_weight = bg_weight * alpha + fg_weight;
  out_color = bg_col * bg_weight * alpha + fg_col * fg_weight;
  out_color *= safe_rcp(out_weight);
}

void main(void)
{
  vec2 uv = uvcoordsvar.xy;

  vec4 noise = texelfetch_noise_tex(gl_FragCoord.xy);
  /* Stochastically randomize which pixel to resolve. This avoids having garbage values
   * from the weight mask interpolation but still have less pixelated look. */
  uv += noise.zw * 0.5 / vec2(textureSize(bgColorBuffer, 0).xy);

  vec4 bg = textureLod(bgColorBuffer, uv, 0.0);
  vec4 fg = textureLod(fgColorBuffer, uv, 0.0);
  float fg_w = textureLod(fgWeightBuffer, uv, 0.0).r;
  float bg_w = textureLod(bgWeightBuffer, uv, 0.0).r;

  ivec2 tile_co = ivec2(gl_FragCoord.xy / 16.0);
  CocTile coc_tile = dof_coc_tile_load(fgTileBuffer, bgTileBuffer, tile_co);

  vec4 focus = vec4(0.0);
  float focus_w = 0.0;
  if (coc_tile.fg_slight_focus_max_coc >= 0.5) {
    dof_slight_focus_gather(coc_tile.fg_slight_focus_max_coc, focus, focus_w);
  }
  else {
    focus = textureLod(fullResColorBuffer, uv, 0.0);
    if (coc_tile.fg_slight_focus_max_coc == DOF_TILE_FOCUS) {
      /* Tile is full in focus. */
      focus_w = 1.0;
    }
    else /* (coc_tile.fg_slight_focus_max_coc == DOF_TILE_DEFOCUS) */ {
      /* Tile is full in defocus. */
      focus_w = 0.0001; /* Almost no weight, used on last resort. */
    }
  }

  /* Composite background. */
  fragColor = bg;
  float weight = float(bg_w > 0.0);

  /* TODO Composite hole filling pass. */

  /* Composite in focus + slight defocus. */
  fragColor = fragColor * (1.0 - focus_w) + focus * focus_w;
  weight = weight * (1.0 - focus_w) + focus_w;
  fragColor *= safe_rcp(weight);

  /* Composite foreground. */
  fragColor = fragColor * (1.0 - fg_w) + fg * fg_w;

#if 0 /* Debug */
  if (coc_tile.fg_slight_focus_max_coc == DOF_TILE_FOCUS) {
    fragColor.rgb *= vec3(1.0, 0.1, 0.1);
  }
#endif
}