void node_composite_hue_correct(float factor,
                                vec4 color,
                                sampler1DArray curve_map,
                                const float layer,
                                vec3 minimums,
                                vec3 range_dividers,
                                out vec4 result)
{
  vec4 hsv;
  rgb_to_hsv(color, hsv);
  vec3 parameters = (hsv.x - minimums) * range_dividers;

  /* A value of 0.5 means no change, so adjust to get an identity at 0.5. */
  hsv.x += texture(curve_map, vec2(parameters.x, layer)).x - 0.5;
  hsv.y *= texture(curve_map, vec2(parameters.y, layer)).y * 2.0;
  hsv.z *= texture(curve_map, vec2(parameters.z, layer)).z * 2.0;

  hsv.x = fract(hsv.x);
  hsv.y = clamp(hsv.y, 0.0, 1.0);

  hsv_to_rgb(hsv, result);

  result = mix(color, result, factor);
}
