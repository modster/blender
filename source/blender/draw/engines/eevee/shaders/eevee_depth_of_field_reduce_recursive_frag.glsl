
/**
 * Reduce recursive pass: Simple coc & luma aware downsampling pass to generate mipmaps.
 **/

#pragma BLENDER_REQUIRE(eevee_depth_of_field_lib.glsl)

uniform sampler2D color_tx;
uniform sampler2D coc_tx;

layout(location = 0) out vec4 out_color;
layout(location = 1) out float out_coc;

/* Downsample pass done for each mip starting from mip1. */
void main()
{
  vec2 input_texel_size = 1.0 / vec2(textureSize(color_tx, 0).xy);
  /* Center uv around the 4 pixels of the previous mip. */
  vec2 quad_center = (floor(gl_FragCoord.xy) * 2.0 + 1.0) * input_texel_size;

  vec4 colors[4];
  vec4 cocs;
  for (int i = 0; i < 4; i++) {
    vec2 sample_uv = quad_center + quad_offsets[i] * input_texel_size;
    colors[i] = textureLod(color_tx, sample_uv, 0.0);
    cocs[i] = textureLod(coc_tx, sample_uv, 0.0).r;
  }

  vec4 weights = dof_bilateral_coc_weights(cocs);
  weights *= dof_bilateral_color_weights(colors);
  /* Normalize so that the sum is 1. */
  weights *= safe_rcp(sum(weights));

  out_color = weighted_sum_array(colors, weights);
  out_coc = dot(cocs, weights);
}