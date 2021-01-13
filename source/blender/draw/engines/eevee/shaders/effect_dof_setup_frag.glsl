
/**
 * Setup pass: CoC and luma aware downsample to half resolution of the input scene color buffer.
 **/

#pragma BLENDER_REQUIRE(effect_dof_lib.glsl)

/* Full resolution. */
uniform sampler2D colorBuffer;
uniform sampler2D depthBuffer;

/* Half resolution. */
layout(location = 0) out vec4 outColor;
layout(location = 1) out float outCoc;

void main()
{
  vec2 fullres_texel_size = 1.0 / vec2(textureSize(colorBuffer, 0).xy);
  /* Center uv around the 4 fullres pixels. */
  vec2 quad_center = (floor(gl_FragCoord.xy) * 2.0 + 1.0) * fullres_texel_size;

  vec4 colors[4];
  vec4 depths;
  for (int i = 0; i < 4; i++) {
    vec2 sample_uv = quad_center + quad_offsets[i] * fullres_texel_size;
    colors[i] = safe_color(textureLod(colorBuffer, sample_uv, 0.0));
    depths[i] = textureLod(depthBuffer, sample_uv, 0.0).r;
  }

  vec4 cocs = dof_coc_from_zdepth(depths);

  vec4 weights = dof_downsample_bilateral_coc_weights(cocs);
  weights *= dof_downsample_bilateral_color_weights(colors);
  /* Normalize so that the sum is 1. */
  weights *= safe_rcp(sum(weights));

  outColor = weighted_sum_array(colors, weights);
  outCoc = dot(cocs, weights);

  outCoc = clamp(outCoc, -bokehMaxsize, bokehMaxsize);
}
