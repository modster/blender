
/**
 * Reduce pass: Downsample the color buffer to generate mipmaps.
 * Also decide if a pixel is to be convolved by scattering or gathering during the first pass.
 **/

#pragma BLENDER_REQUIRE(effect_dof_lib.glsl)

/** Inputs:
 * COPY_PASS: Is output of setup pass (halfres).
 * REDUCE_PASS: Is previous Gather input miplvl (halfres >> miplvl).
 **/
uniform sampler2D colorBuffer;
uniform sampler2D cocBuffer;

/** Outputs:
 * COPY_PASS: Gather input mip0.
 * REDUCE_PASS: Is next Gather input miplvl (halfres >> miplvl).
 **/
layout(location = 0) out vec4 outColor;
layout(location = 1) out float outCoc;

#ifdef COPY_PASS

/* Simple copy pass where we select what pixels to scatter. Also the resolution might change.
 * NOTE: The texture can end up being too big because of the mipmap padding. We correct for that
 * during the convolution phase. */
void main()
{
  vec2 uv = gl_FragCoord.xy / vec2(textureSize(colorBuffer, 0).xy);

  outColor = textureLod(colorBuffer, uv, 0.0);
  outCoc = textureLod(cocBuffer, uv, 0.0).r;

  /* Only scatter if luminous enough. */
  bool do_scatter = any(greaterThan(outColor.rgb, vec3(scatterColorThreshold)));
  /* Only scatter if CoC is big enough. */
  do_scatter = do_scatter && (abs(outCoc) > scatterCocThreshold);
  /* TODO(fclem) Needs the downsample pass. */
  // do_scatter = do_scatter && dof_scatter_neighborhood_rejection(outColor, uv);

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
  /* Normalize so that the sum is 1. */
  weights *= safe_rcp(sum(weights));

  outColor = weighted_sum_array(colors, weights);
  outCoc = dot(cocs, weights);
}

#endif
