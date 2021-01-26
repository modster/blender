
/**
 * Reduce pass: Downsample the color buffer to generate mipmaps.
 * Also decide if a pixel is to be convolved by scattering or gathering during the first pass.
 **/

#pragma BLENDER_REQUIRE(effect_dof_lib.glsl)

/** Inputs:
 * COPY_PASS: Is output of setup pass (halfres) and downsample pass (quarter res).
 * REDUCE_PASS: Is previous Gather input miplvl (halfres >> miplvl).
 **/
uniform sampler2D colorBuffer;
uniform sampler2D cocBuffer;
uniform sampler2D downsampledBuffer;

uniform float bokehRatio;
uniform float scatterColorThreshold;
uniform float scatterCocThreshold;
uniform float scatterColorNeighborMax;

/** Outputs:
 * COPY_PASS: Gather input mip0.
 * REDUCE_PASS: Is next Gather input miplvl (halfres >> miplvl).
 **/
layout(location = 0) out vec4 outColor;
layout(location = 1) out float outCoc;

#ifdef COPY_PASS

layout(location = 2) out vec3 outScatterColor;

/* NOTE: Do not compare alpha as it is not scattered by the scatter pass. */
float dof_scatter_neighborhood_rejection(vec3 color)
{
  color = min(vec3(scatterColorNeighborMax), color);

  float validity = 0.0;

  /* Centered in the middle of 4 quarter res texel. */
  vec2 texel_size = 1.0 / vec2(textureSize(downsampledBuffer, 0).xy);
  vec2 uv = (floor(gl_FragCoord.xy * 0.5) + 1.0) * texel_size;

  vec3 max_diff = vec3(0.0);
  for (int i = 0; i < 4; i++) {
    vec2 sample_uv = uv + 2.0 * quad_offsets[i] * texel_size;
    vec3 ref = textureLod(downsampledBuffer, sample_uv, 0.0).rgb;

    ref = min(vec3(scatterColorNeighborMax), ref);
    float diff = max_v3(max(vec3(0.0), ref - color));

    const float rejection_threshold = 0.7;
    diff = saturate(diff / rejection_threshold - 1.0);
    validity = max(validity, diff);
  }

  return validity;
}

/* This avoids sprite popping in and out at the screen border and
 * drawing sprites larger than the screen. */
float dof_scatter_screen_border_rejection(float coc, vec2 uv, vec2 screen_size)
{
  vec2 screen_pos = uv * screen_size;
  float min_screen_border_distance = min_v2(min(screen_pos, screen_size - screen_pos));
  /* Fullres to halfres CoC. */
  coc *= 0.5;
  /* Allow 10px transition. */
  const float rejection_hardeness = 1.0 / 10.0;
  return saturate((min_screen_border_distance - abs(coc)) * rejection_hardeness + 1.0);
}

float dof_scatter_luminosity_rejection(vec3 color)
{
  const float rejection_hardness = 5.0;
  return saturate(max_v3(color - scatterColorThreshold) * rejection_hardness);
}

float dof_scatter_coc_radius_rejection(float coc)
{
  const float rejection_hardness = 0.3;
  return saturate((abs(coc) - scatterCocThreshold) * rejection_hardness);
}

/* Simple copy pass where we select what pixels to scatter. Also the resolution might change.
 * NOTE: The texture can end up being too big because of the mipmap padding. We correct for
 * that during the convolution phase. */
void main()
{
  vec2 halfres = vec2(textureSize(colorBuffer, 0).xy);
  vec2 uv = gl_FragCoord.xy / halfres;

  outColor = textureLod(colorBuffer, uv, 0.0);
  outCoc = textureLod(cocBuffer, uv, 0.0).r;

  /* Only scatter if luminous enough. */
  float do_scatter = dof_scatter_luminosity_rejection(outColor.rgb);
  /* Only scatter if CoC is big enough. */
  do_scatter *= dof_scatter_coc_radius_rejection(outCoc);
  /* Only scatter if CoC is not too big to avoid performance issues. */
  do_scatter *= dof_scatter_screen_border_rejection(outCoc, uv, halfres);
  /* Only scatter if neighborhood is different enough. Test is expensive, do only if worth it. */
  do_scatter *= (do_scatter < 0.2) ? 1.0 : dof_scatter_neighborhood_rejection(outColor.rgb);
  /* For debuging. */
  do_scatter *= float(!no_scatter_pass);
  /* Sharpen. */
  do_scatter *= do_scatter * do_scatter;

  outScatterColor = mix(vec3(0.0), outColor.rgb, do_scatter);
  outColor.rgb = mix(outColor.rgb, vec3(0.0), do_scatter);

  /* Apply energy conservation to anamorphic scattered bokeh. */
  outScatterColor /= bokehRatio;
}

#else /* REDUCE_PASS */

/* Downsample pass done for each mip starting from mip1. */
void main()
{
  vec2 input_texel_size = 1.0 / vec2(textureSize(colorBuffer, 0).xy);
  /* Center uv around the 4 pixels of the previous mip. */
  vec2 quad_center = (floor(gl_FragCoord.xy) * 2.0 + 1.0) * input_texel_size;

  vec4 colors[4];
  vec4 cocs;
  for (int i = 0; i < 4; i++) {
    vec2 sample_uv = quad_center + quad_offsets[i] * input_texel_size;
    colors[i] = dof_load_gather_color(colorBuffer, sample_uv, 0.0);
    cocs[i] = textureLod(cocBuffer, sample_uv, 0.0).r;
  }

  vec4 weights = dof_downsample_bilateral_coc_weights(cocs);
  weights *= dof_downsample_bilateral_color_weights(colors);
  /* Normalize so that the sum is 1. */
  weights *= safe_rcp(sum(weights));

  outColor = weighted_sum_array(colors, weights);
  outCoc = dot(cocs, weights);
}

#endif
