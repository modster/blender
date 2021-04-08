
/**
 * Setup pass: CoC and luma aware downsample to half resolution of the input scene color buffer.
 *
 * An addition to the downsample CoC, we output the maximum slight out of focus CoC to be
 * sure we don't miss a pixel.
 *
 * Input:
 *  Full-resolution color & depth buffer
 * Output:
 *  Half-resolution Color, signed CoC (out_coc.x), and max slight focus abs CoC (out_coc.y).
 **/

#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_depth_of_field_lib.glsl)

layout(std140) uniform dof_block
{
  DepthOfFieldData dof;
};

uniform sampler2D color_tx;
uniform sampler2D depth_tx;

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec2 out_coc;

float dof_abs_max_slight_of_focus_coc(vec4 cocs)
{
  /* Clamp to 0.5 if full in defocus to differentiate full focus tiles with coc == 0.0.
   * This enables an optimization in the resolve pass. */
  const vec4 threshold = vec4(dof_layer_threshold + dof_layer_offset);
  cocs = abs(cocs);
  bvec4 defocus = greaterThan(cocs, threshold);
  bvec4 focus = lessThanEqual(cocs, vec4(0.5));
  if (any(defocus) && any(focus)) {
    /* For the same reason as in the flatten pass. This is a case we cannot optimize for. */
    cocs = mix(cocs, vec4(dof_tile_mixed), focus);
    cocs = mix(cocs, vec4(dof_tile_mixed), defocus);
  }
  else {
    cocs = mix(cocs, vec4(dof_tile_focus), focus);
    cocs = mix(cocs, vec4(dof_tile_defocus), defocus);
  }
  return max_v4(cocs);
}

void main()
{
  vec2 fullres_texel_size = 1.0 / vec2(textureSize(color_tx, 0).xy);
  /* Center uv around the 4 fullres pixels. */
  vec2 quad_center = (floor(gl_FragCoord.xy) * 2.0 + 1.0) * fullres_texel_size;

  vec4 colors[4];
  vec4 cocs;
  for (int i = 0; i < 4; i++) {
    vec2 sample_uv = quad_center + quad_offsets[i] * fullres_texel_size;
    colors[i] = safe_color(textureLod(color_tx, sample_uv, 0.0));
    cocs[i] = dof_coc_from_depth(dof, sample_uv, textureLod(depth_tx, sample_uv, 0.0).r);
  }

  cocs = clamp(cocs, -dof.coc_abs_max, dof.coc_abs_max);

  vec4 weights = dof_bilateral_coc_weights(cocs);
  weights *= dof_bilateral_color_weights(colors);
  /* Normalize so that the sum is 1. */
  weights *= safe_rcp(sum(weights));

  out_color = weighted_sum_array(colors, weights);
  out_coc.x = dot(cocs, weights);
  out_coc.y = dof_abs_max_slight_of_focus_coc(cocs);
}
