
#pragma BLENDER_REQUIRE(effect_dof_lib.glsl)

flat in vec4 color1;
flat in vec4 color2;
flat in vec4 color3;
flat in vec4 color4;
flat in vec4 weights;
flat in vec4 cocs;
flat in vec2 spritepos;

layout(location = 0) out vec4 fragColor;

float bokeh_shape(vec2 center)
{
  vec2 co = gl_FragCoord.xy - center;
  float dist = length(co);

#if 0
  if (bokeh_sides.x > 0.0) {
    /* Circle parametrization */
    float theta = atan(co.y, co.x) + bokeh_rotation;
    /* Optimized version of :
     * float denom = theta - (M_2PI / bokeh_sides) * floor((bokeh_sides * theta + M_PI) / M_2PI);
     * float r = cos(M_PI / bokeh_sides) / cos(denom); */
    float denom = theta - bokeh_sides.y * floor(bokeh_sides.z * theta + 0.5);
    float r = bokeh_sides.w / cos(denom);
    /* Divide circle radial coord by the shape radius for angle theta.
     * Giving us the new linear radius to the shape edge. */
    dist /= r;
  }
#endif

  return dist;
}

/* accumulate color in the near/far blur buffers */
void main(void)
{
  vec4 shapes;
  for (int i = 0; i < 4; i++) {
    shapes[i] = bokeh_shape(spritepos + quad_offsets[i]);
  }
  /* Becomes signed distance field in pixel units. */
  shapes -= cocs;
  /* Smooth the edges a bit to fade out the undersampling artifacts. */
  shapes = 1.0 - smoothstep(-0.6, 0.6, shapes);
  /* Outside of bokeh shape. Try to avoid overloading ROPs. */
  if (max_v4(shapes) == 0.0) {
    discard;
  }

  fragColor = color1 * shapes.x;
  fragColor += color2 * shapes.y;
  fragColor += color3 * shapes.z;
  fragColor += color4 * shapes.w;

  /* Do not accumulate alpha. This has already been accumulated by the gather pass. */
  fragColor.a = 0.0;
}
