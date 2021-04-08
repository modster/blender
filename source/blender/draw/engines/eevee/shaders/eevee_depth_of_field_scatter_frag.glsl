
/**
 * Scatter pass: Use sprites to scatter the color of very bright pixel to have higher quality blur.
 *
 * We only scatter one triangle per sprite and one sprite per 4 pixels to reduce vertex shader
 * invocations and overdraw.
 **/

#pragma BLENDER_REQUIRE(eevee_depth_of_field_scatter_lib.glsl)

layout(std140) uniform dof_block
{
  DepthOfFieldData dof;
};

uniform sampler2D occlusion_tx;
uniform sampler2D bokeh_lut_tx;

layout(location = 0) out vec4 fragColor;

float bokeh_shape(vec2 center)
{
  vec2 co = gl_FragCoord.xy - center;
  float dist;
  if (DOF_BOKEH_TEXTURE) {
    if (DOF_FOREGROUND_PASS) {
      /* Negate to flip bokeh shape. Mimics optical phenomenon. */
      co = -co;
    }
    co *= dof.bokeh_anisotropic_scale_inv;
    float texture_size = float(textureSize(bokeh_lut_tx, 0).x);
    /* Bias scale to avoid sampling at the texture's border. */
    float scale_fac = spritesize * (float(DOF_BOKEH_LUT_SIZE) / float(DOF_BOKEH_LUT_SIZE - 1));
    dist = scale_fac * textureLod(bokeh_lut_tx, (co / scale_fac) * 0.5 + 0.5, 0.0).r;
  }
  else {
    dist = length(co);
  }
  return dist;
}

#define linearstep(p0, p1, v) (clamp(((v) - (p0)) / abs((p1) - (p0)), 0.0, 1.0))

void main(void)
{
  vec4 shapes;
  for (int i = 0; i < 4; i++) {
    shapes[i] = bokeh_shape(spritepos + quad_offsets[i]);
  }
  /* Becomes signed distance field in pixel units. */
  shapes -= cocs;
  /* Smooth the edges a bit to fade out the undersampling artifacts. */
  shapes = 1.0 - linearstep(-0.8, 0.8, shapes);
  /* Outside of bokeh shape. Try to avoid overloading ROPs. */
  if (max_v4(shapes) == 0.0) {
    discard;
  }

  if (!no_scatter_occlusion) {
    /* Works because target is the same size as occlusion_tx. */
    vec2 uv = gl_FragCoord.xy / vec2(textureSize(occlusion_tx, 0).xy);
    vec2 occlusion_data = texture(occlusion_tx, uv).rg;
    /* Fix tilling artifacts. (Slide 90) */
    const float correction_fac = 1.0 - DOF_FAST_GATHER_COC_ERROR;
    /* Occlude the sprite with geometry from the same field
     * using a VSM like chebychev test (slide 85). */
    float mean = occlusion_data.x;
    float variance = occlusion_data.x;
    shapes *= variance * safe_rcp(variance + sqr(max(cocs * correction_fac - mean, 0.0)));
  }

  fragColor = color1 * shapes.x;
  fragColor += color2 * shapes.y;
  fragColor += color3 * shapes.z;
  fragColor += color4 * shapes.w;

  /* Do not accumulate alpha. This has already been accumulated by the gather pass. */
  fragColor.a = 0.0;

  if (debug_scatter_perf) {
    fragColor.rgb = avg(fragColor.rgb) * vec3(1.0, 0.0, 0.0);
  }
}
