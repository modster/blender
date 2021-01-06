

#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)

uniform sampler2D colorBuffer;
uniform sampler2D depthBuffer;

uniform vec2 dofParams;
uniform bool unpremult;

#define dof_mul dofParams.x /* distance * aperturesize * invsensorsize */
#define dof_bias dofParams.y /* aperturesize * invsensorsize */

uniform vec4 bokehParams[2];

#define bokeh_rotation bokehParams[0].x
#define bokeh_ratio bokehParams[0].y
#define bokeh_maxsize bokehParams[0].z
#define bokeh_sides \
  bokehParams[1] /* Polygon Bokeh shape number of sides (with precomputed vars) */

uniform vec2 nearFar; /* Near & far view depths values */

/* -------------- Utils ------------- */

/* divide by sensor size to get the normalized size */
#define calculate_coc(zdepth) (dof_mul / zdepth - dof_bias)

#define linear_depth(z) \
  ((ProjectionMatrix[3][3] == 0.0) ? \
       (nearFar.x * nearFar.y) / (z * (nearFar.x - nearFar.y) + nearFar.y) : \
       z * (nearFar.y - nearFar.x) + nearFar.x) /* Only true for camera view! */

vec4 safe_color(vec4 c)
{
  /* Clamp to avoid black square artifacts if a pixel goes NaN. */
  return clamp(c, vec4(0.0), vec4(1e20)); /* 1e20 arbitrary. */
}

#define THRESHOLD 1.0

#ifdef STEP_DOWNSAMPLE

layout(location = 0) out vec4 nearColor;
layout(location = 1) out vec4 farColor;
layout(location = 2) out vec2 cocData;

/* Downsample the color buffer to half resolution.
 * Weight color samples by
 * Compute maximum CoC for near and far blur. */
void main(void)
{
  ivec4 uvs = ivec4(gl_FragCoord.xyxy) * 2 + ivec4(0, 0, 1, 1);

  /* custom downsampling */
  vec4 color1 = safe_color(texelFetch(colorBuffer, uvs.xy, 0));
  vec4 color2 = safe_color(texelFetch(colorBuffer, uvs.zw, 0));
  vec4 color3 = safe_color(texelFetch(colorBuffer, uvs.zy, 0));
  vec4 color4 = safe_color(texelFetch(colorBuffer, uvs.xw, 0));

  /* Leverage SIMD by combining 4 depth samples into a vec4 */
  vec4 depth;
  depth.r = texelFetch(depthBuffer, uvs.xy, 0).r;
  depth.g = texelFetch(depthBuffer, uvs.zw, 0).r;
  depth.b = texelFetch(depthBuffer, uvs.zy, 0).r;
  depth.a = texelFetch(depthBuffer, uvs.xw, 0).r;

  vec4 zdepth = linear_depth(depth);

  /* Compute signed CoC for each depth samples */
  vec4 coc_near = calculate_coc(zdepth);
  vec4 coc_far = -coc_near;

  cocData.x = max(max_v4(coc_near), 0.0);
  cocData.y = max(max_v4(coc_far), 0.0);

  /* now we need to write the near-far fields premultiplied by the coc
   * also use bilateral weighting by each coc values to avoid bleeding. */
  vec4 near_weights = step(THRESHOLD, coc_near) * clamp(1.0 - abs(cocData.x - coc_near), 0.0, 1.0);
  vec4 far_weights = step(THRESHOLD, coc_far) * clamp(1.0 - abs(cocData.y - coc_far), 0.0, 1.0);

#  ifdef USE_ALPHA_DOF
  /* Premult */
  color1.rgb *= color1.a;
  color2.rgb *= color2.a;
  color3.rgb *= color3.a;
  color4.rgb *= color4.a;
#  endif

  /* now write output to weighted buffers. */
  nearColor = weighted_sum(color1, color2, color3, color4, near_weights);
  farColor = weighted_sum(color1, color2, color3, color4, far_weights);
}

#elif defined(STEP_SCATTER)

flat in vec4 color1;
flat in vec4 color2;
flat in vec4 color3;
flat in vec4 color4;
flat in vec4 weights;
flat in vec4 cocs;
flat in vec2 spritepos;

layout(location = 0) out vec4 fragColor;
#  ifdef USE_ALPHA_DOF
layout(location = 1) out float fragAlpha;
#  endif

float bokeh_shape(vec2 center)
{
  vec2 co = gl_FragCoord.xy - center;
  float dist = length(co);

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

  return dist;
}

/* accumulate color in the near/far blur buffers */
void main(void)
{
  vec4 shapes;
  shapes.x = bokeh_shape(spritepos + vec2(-0.5, 0.5));
  shapes.y = bokeh_shape(spritepos + vec2(0.5, 0.5));
  shapes.z = bokeh_shape(spritepos + vec2(0.5, -0.5));
  shapes.w = bokeh_shape(spritepos + vec2(-0.5, -0.5));
  /* Becomes signed distance field in pixel units. */
  shapes -= cocs;
  /* Smooth the edges a bit. This effectively soften the bokeh shape
   * but does fade out the undersampling artifacts. */
  vec4 fac = clamp(cocs - 1.0, 0.0, 1.0) * 0.6 + 0.01;
  shapes = 1.0 - smoothstep(-fac, fac, shapes);
  /* Outside of bokeh shape. Try to avoid overloading ROPs. */
  if (max_v4(shapes) == 0.0) {
    discard;
  }

  fragColor = weighted_sum(color1, color2, color3, color4, shapes);

#  ifdef USE_ALPHA_DOF
  fragAlpha = fragColor.a;
  fragColor.a = sum(weights * shapes);
#  endif
}

#elif defined(STEP_RESOLVE)

#  define MERGE_THRESHOLD 4.0

uniform sampler2D scatterNearBuffer;
uniform sampler2D scatterNearAlphaBuffer;
uniform sampler2D scatterFarBuffer;
uniform sampler2D scatterFarAlphaBuffer;

in vec4 uvcoordsvar;
out vec4 fragColor;

vec4 upsample_filter(sampler2D tex, vec2 uv, vec2 texelSize)
{
#  if 1 /* 9-tap bilinear upsampler (tent filter) */
  vec4 d = texelSize.xyxy * vec4(1, 1, -1, 0);

  vec4 s;
  s = textureLod(tex, uv - d.xy, 0.0);
  s += textureLod(tex, uv - d.wy, 0.0) * 2;
  s += textureLod(tex, uv - d.zy, 0.0);

  s += textureLod(tex, uv + d.zw, 0.0) * 2;
  s += textureLod(tex, uv, 0.0) * 4;
  s += textureLod(tex, uv + d.xw, 0.0) * 2;

  s += textureLod(tex, uv + d.zy, 0.0);
  s += textureLod(tex, uv + d.wy, 0.0) * 2;
  s += textureLod(tex, uv + d.xy, 0.0);

  return s * (1.0 / 16.0);
#  else
  /* 4-tap bilinear upsampler */
  vec4 d = texelSize.xyxy * vec4(-1, -1, +1, +1) * 0.5;

  vec4 s;
  s = textureLod(tex, uv + d.xy, 0.0);
  s += textureLod(tex, uv + d.zy, 0.0);
  s += textureLod(tex, uv + d.xw, 0.0);
  s += textureLod(tex, uv + d.zw, 0.0);

  return s * (1.0 / 4.0);
#  endif
}

/* Combine the Far and Near color buffers */
void main(void)
{
  vec2 uv = uvcoordsvar.xy;
  /* Recompute Near / Far CoC per pixel */
  float depth = textureLod(depthBuffer, uv, 0.0).r;
  float zdepth = linear_depth(depth);
  float coc_signed = calculate_coc(zdepth);
  float coc_far = max(-coc_signed, 0.0);
  float coc_near = max(coc_signed, 0.0);

  vec4 focus_col = textureLod(colorBuffer, uv, 0.0);

  vec2 texelSize = 1.0 / vec2(textureSize(scatterNearBuffer, 0));
  vec4 near_col = upsample_filter(scatterNearBuffer, uv, texelSize);
  vec4 far_col = upsample_filter(scatterFarBuffer, uv, texelSize);

  float far_w = far_col.a;
  float near_w = near_col.a;
  float focus_w = 1.0 - smoothstep(1.0, MERGE_THRESHOLD, abs(coc_signed));
  float inv_weight_sum = 1.0 / (near_w + focus_w + far_w);

  focus_col *= focus_w; /* Premul */

#  ifdef USE_ALPHA_DOF
  near_col.a = upsample_filter(scatterNearAlphaBuffer, uv, texelSize).r;
  far_col.a = upsample_filter(scatterFarAlphaBuffer, uv, texelSize).r;
#  endif

  fragColor = (far_col + near_col + focus_col) * inv_weight_sum;

#  ifdef USE_ALPHA_DOF
  /* Sigh... viewport expect premult output but
   * the final render output needs to be with
   * associated alpha. */
  if (unpremult) {
    fragColor.rgb /= (fragColor.a > 0.0) ? fragColor.a : 1.0;
  }
#  endif
}

#endif
