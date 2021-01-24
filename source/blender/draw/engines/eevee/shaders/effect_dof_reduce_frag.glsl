
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

/** Outputs:
 * COPY_PASS: Gather input mip0.
 * REDUCE_PASS: Is next Gather input miplvl (halfres >> miplvl).
 **/
layout(location = 0) out vec4 outColor;
layout(location = 1) out float outCoc;

#ifdef COPY_PASS

/* TODO(fclem) Output scatter color to a separate R11G11B10 buffer. */
// layout(location = 2) out float outScatterColor;

vec3 non_linear_comparison_space(vec3 color)
{
  /* TODO(fclem) we might want something more aware of exposure. */
  return -1.0 / (-1.0 - max(vec3(0.0), (color - scatterColorThreshold)));
}

/* NOTE: Do not compare alpha as it is not scattered by the scatter pass. */
bool dof_scatter_neighborhood_rejection(vec3 color)
{
  color = non_linear_comparison_space(color);
  /* Centered in the middle of 4 quarter res texel. */
  vec2 texel_size = 1.0 / vec2(textureSize(downsampledBuffer, 0).xy);
  vec2 uv = (floor(gl_FragCoord.xy * 0.5) + 1.0) * texel_size;

  vec3 max_diff = vec3(0.0);
  for (int i = 0; i < 4; i++) {
    vec2 sample_uv = uv + 2.0 * quad_offsets[i] * texel_size;
    vec3 ref = textureLod(downsampledBuffer, sample_uv, 0.0).rgb;
    ref = non_linear_comparison_space(ref);
    max_diff = max(max_diff, abs(ref - color));
  }
  /* TODO(fclem) Adjust using multiple test scene. */
  const float rejection_threshold = 0.05;
  bool valid = max_v3(max_diff) > rejection_threshold;

  return valid;
}

/* Simple copy pass where we select what pixels to scatter. Also the resolution might change.
 * NOTE: The texture can end up being too big because of the mipmap padding. We correct for
 * that during the convolution phase. */
void main()
{
  vec2 uv = gl_FragCoord.xy / vec2(textureSize(colorBuffer, 0).xy);

  outColor = textureLod(colorBuffer, uv, 0.0);
  outCoc = textureLod(cocBuffer, uv, 0.0).r;

  /* Only scatter if luminous enough. */
  bool do_scatter = any(greaterThan(outColor.rgb, vec3(scatterColorThreshold)));
  /* Only scatter if CoC is big enough. */
  do_scatter = do_scatter && (abs(outCoc) > scatterCocThreshold);
  /* Only scatter if CoC is not too big to avoid performance issues. */
  do_scatter = do_scatter && (abs(outCoc) < 200.0); /* TODO(fclem) user threshold. */
  /* Only scatter if neighborhood is different enough. */
  do_scatter = do_scatter && dof_scatter_neighborhood_rejection(outColor.rgb);
  /* For debuging. */
  do_scatter = !no_scatter_pass && do_scatter;

  /**
   * NOTE: Here we deviate from the reference implementation. Since we cannot write a sprite list
   * directly (because of minimal hardware restriction), we keep the pixel intensity the same for
   * mip0 and negate the color for scattered pixels.
   * This make sure we can get back the color for each convolution method and still select
   * only some pixels for scattering. However this does not give us the possibility to have a
   * smooth transition between the two methods.
   **/
  if (do_scatter) {
    /* Negate the color to specify that we want it to be scattered. */
    outColor.rgb *= -1.0;
  }
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
