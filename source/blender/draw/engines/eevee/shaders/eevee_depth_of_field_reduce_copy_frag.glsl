
/**
 * Reduce copy pass: filter fireflies and split color between scatter and gather input.
 *
 * NOTE: The texture can end up being too big because of the mipmap padding. We correct for
 * that during the convolution phase.
 *
 * Inputs:
 * - Output of setup pass (halfres) and reduce downsample pass (quarter res).
 * Outputs:
 * - Halfres padded to avoid mipmap mis-alignment (so possibly not matching input size).
 * - Gather input color (mip 0), Scatter input color, Signed CoC.
 **/

#pragma BLENDER_REQUIRE(eevee_depth_of_field_lib.glsl)

layout(std140) uniform dof_block
{
  DepthOfFieldData dof;
};

uniform sampler2D color_tx;
uniform sampler2D coc_tx;
uniform sampler2D downsampled_tx;

layout(location = 0) out vec4 out_color_gather;
layout(location = 1) out float out_coc;
layout(location = 2) out vec3 out_color_scatter;

/* NOTE: Do not compare alpha as it is not scattered by the scatter pass. */
float dof_scatter_neighborhood_rejection(vec3 color)
{
  color = min(vec3(dof.scatter_neighbor_max_color), color);

  float validity = 0.0;

  /* Centered in the middle of 4 quarter res texel. */
  vec2 texel_size = 1.0 / vec2(textureSize(downsampled_tx, 0).xy);
  vec2 uv = (gl_FragCoord.xy * 0.5) * texel_size;

  vec3 max_diff = vec3(0.0);
  for (int i = 0; i < 4; i++) {
    vec2 sample_uv = uv + quad_offsets[i] * texel_size;
    vec3 ref = textureLod(downsampled_tx, sample_uv, 0.0).rgb;

    ref = min(vec3(dof.scatter_neighbor_max_color), ref);
    float diff = max_v3(max(vec3(0.0), abs(ref - color)));

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
  const float rejection_hardness = 1.0;
  return saturate(max_v3(color - dof.scatter_color_threshold) * rejection_hardness);
}

float dof_scatter_coc_radius_rejection(float coc)
{
  const float rejection_hardness = 0.3;
  return saturate((abs(coc) - dof.scatter_coc_threshold) * rejection_hardness);
}

float fast_luma(vec3 color)
{
  return (2.0 * color.g) + color.r + color.b;
}

/* Lightweight version of neighborhood clamping found in TAA. */
vec3 dof_neighborhood_clamping(vec3 color)
{
  vec2 texel_size = 1.0 / vec2(textureSize(color_tx, 0));
  vec2 uv = gl_FragCoord.xy * texel_size;
  vec4 ofs = vec4(-1, 1, -1, 1) * texel_size.xxyy;

  /* Luma clamping. 3x3 square neighborhood. */
  float c00 = fast_luma(textureLod(color_tx, uv + ofs.xz, 0.0).rgb);
  float c01 = fast_luma(textureLod(color_tx, uv + ofs.xz * vec2(1.0, 0.0), 0.0).rgb);
  float c02 = fast_luma(textureLod(color_tx, uv + ofs.xw, 0.0).rgb);

  float c10 = fast_luma(textureLod(color_tx, uv + ofs.xz * vec2(0.0, 1.0), 0.0).rgb);
  float c11 = fast_luma(color);
  float c12 = fast_luma(textureLod(color_tx, uv + ofs.xw * vec2(0.0, 1.0), 0.0).rgb);

  float c20 = fast_luma(textureLod(color_tx, uv + ofs.yz, 0.0).rgb);
  float c21 = fast_luma(textureLod(color_tx, uv + ofs.yz * vec2(1.0, 0.0), 0.0).rgb);
  float c22 = fast_luma(textureLod(color_tx, uv + ofs.yw, 0.0).rgb);

  float avg_luma = avg8(c00, c01, c02, c10, c12, c20, c21, c22);
  float max_luma = max8(c00, c01, c02, c10, c12, c20, c21, c22);

  float upper_bound = mix(max_luma, avg_luma, dof.denoise_factor);
  upper_bound = mix(c11, upper_bound, dof.denoise_factor);

  float clamped_luma = min(upper_bound, c11);

  return color * clamped_luma * safe_rcp(c11);
}

void main()
{
  vec2 halfres = vec2(textureSize(color_tx, 0).xy);
  vec2 uv = gl_FragCoord.xy / halfres;

  out_color_gather = textureLod(color_tx, uv, 0.0);
  out_coc = textureLod(coc_tx, uv, 0.0).r;

  out_color_gather.rgb = dof_neighborhood_clamping(out_color_gather.rgb);

  /* Only scatter if luminous enough. */
  float do_scatter = dof_scatter_luminosity_rejection(out_color_gather.rgb);
  /* Only scatter if CoC is big enough. */
  do_scatter *= dof_scatter_coc_radius_rejection(out_coc);
  /* Only scatter if CoC is not too big to avoid performance issues. */
  do_scatter *= dof_scatter_screen_border_rejection(out_coc, uv, halfres);
  /* Only scatter if neighborhood is different enough. */
  do_scatter *= dof_scatter_neighborhood_rejection(out_color_gather.rgb);
  /* For debuging. */
  do_scatter *= float(!no_scatter_pass);

  out_color_scatter = mix(vec3(0.0), out_color_gather.rgb, do_scatter);
  out_color_gather.rgb = mix(out_color_gather.rgb, vec3(0.0), do_scatter);

  /* Apply energy conservation to anamorphic scattered bokeh. */
  out_color_scatter *= max_v2(dof.bokeh_anisotropic_scale_inv);
}
