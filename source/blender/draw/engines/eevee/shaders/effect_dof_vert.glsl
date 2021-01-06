
#pragma BLENDER_REQUIRE(common_math_lib.glsl)

uniform bool nearPass;
uniform vec4 bokehParams[2];

#define bokeh_rotation bokehParams[0].x
#define bokeh_ratio bokehParams[0].y
#define bokeh_maxsize bokehParams[0].z

uniform sampler2D colorBuffer;
uniform sampler2D cocBuffer;

/* Scatter pass, calculate a triangle covering the CoC.
 * We render to a half resolution target with double width so we can
 * separate near and far fields. We also generate only one triangle per group of 4 pixels
 * to limit overdraw. */

flat out vec4 color1;
flat out vec4 color2;
flat out vec4 color3;
flat out vec4 color4;
flat out vec4 weights;
flat out vec4 cocs;
flat out vec2 spritepos;

/* Load 4 Circle of confusion values. texel_co is centered around the 4 taps. */
vec4 fetch_cocs(vec2 texel_co)
{
  /* TODO(fclem) The textureGather(sampler, co, comp) variant isn't here on some implementations.*/
#if 0  // GPU_ARB_texture_gather
  vec2 uvs = texel_co / vec2(textureSize(cocBuffer, 0));
  /* Reminder: Samples order is CW starting from top left. */
  cocs = textureGather(cocBuffer, uvs, nearPass ? 0 : 1);
#else
  ivec2 texel = ivec2(texel_co - 0.5);
  vec4 cocs;
  if (nearPass) {
    cocs.x = texelFetchOffset(cocBuffer, texel, 0, ivec2(0, 1)).r;
    cocs.y = texelFetchOffset(cocBuffer, texel, 0, ivec2(1, 1)).r;
    cocs.z = texelFetchOffset(cocBuffer, texel, 0, ivec2(1, 0)).r;
    cocs.w = texelFetchOffset(cocBuffer, texel, 0, ivec2(0, 0)).r;
  }
  else {
    cocs.x = texelFetchOffset(cocBuffer, texel, 0, ivec2(0, 1)).g;
    cocs.y = texelFetchOffset(cocBuffer, texel, 0, ivec2(1, 1)).g;
    cocs.z = texelFetchOffset(cocBuffer, texel, 0, ivec2(1, 0)).g;
    cocs.w = texelFetchOffset(cocBuffer, texel, 0, ivec2(0, 0)).g;
  }
#endif
  /* We are scattering at half resolution, so divide CoC by 2. */
  return cocs * 0.5;
}

void main()
{
  ivec2 tex_size = textureSize(cocBuffer, 0);
  /* We render to a double width texture so compute
   * the target texel size accordingly */
  vec2 texel_size = 1.0 / vec2(tex_size);

  int t_id = gl_VertexID / 3; /* Triangle Id */

  /* Some math to get the target pixel. */
  int half_tex_width = tex_size.x / 2;
  ivec2 texelco = ivec2(t_id % half_tex_width, t_id / half_tex_width) * 2;

  /* Center sprite around the 4 texture taps. */
  spritepos = vec2(texelco) + 1.0;

  cocs = fetch_cocs(spritepos);

  /* Clamp to max size for performance. */
  /* TODO. Maybe clamp Circle of confusion radius during downsample pass. */
  cocs = min(cocs, bokeh_maxsize);
  float max_coc = max_v4(cocs);

  if (max_coc >= 0.5) {
    /* find the area the pixel will cover and divide the color by it */
    weights = 1.0 / (cocs * cocs * M_PI);
    weights = mix(vec4(0.0), weights, greaterThanEqual(cocs, vec4(0.5)));

    color1 = texelFetchOffset(colorBuffer, texelco, 0, ivec2(0, 1)) * weights.x;
    color2 = texelFetchOffset(colorBuffer, texelco, 0, ivec2(1, 1)) * weights.y;
    color3 = texelFetchOffset(colorBuffer, texelco, 0, ivec2(1, 0)) * weights.z;
    color4 = texelFetchOffset(colorBuffer, texelco, 0, ivec2(0, 0)) * weights.w;
  }
  else {
    /* Don't produce any fragments */
    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
    color1 = color2 = color3 = color4 = vec4(0.0);
    return;
  }

  /* Generate Triangle : less memory fetches from a VBO */
  int v_id = gl_VertexID % 3; /* Vertex Id */

  /* Extend to cover at least the unit circle */
  const float extend = (cos(M_PI / 4.0) + 1.0) * 2.0;
  /* Crappy diagram
   * ex 1
   *    | \
   *    |   \
   *  1 |     \
   *    |       \
   *    |         \
   *  0 |     x     \
   *    |   Circle    \
   *    |   Origin      \
   * -1 0 --------------- 2
   *   -1     0     1     ex
   */
  gl_Position.x = float(v_id / 2) * extend - 1.0; /* int divisor round down */
  gl_Position.y = float(v_id % 2) * extend - 1.0;
  gl_Position.z = 0.0;
  gl_Position.w = 1.0;

  /* Add 1 to max_coc because the max_coc may not be centered on the sprite origin. */
  gl_Position.xy *= (max_coc + 1.0) * vec2(bokeh_ratio, 1.0);
  /* Position the sprite. */
  gl_Position.xy += spritepos;
  /* NDC range [-1..1]. */
  gl_Position.xy = gl_Position.xy * texel_size * 2.0 - 1.0;
}
