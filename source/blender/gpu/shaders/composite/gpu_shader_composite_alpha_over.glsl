
void node_composite_alpha_over(
    float fac, vec4 color1, vec4 color2, float premult_convert, float premult_fac, out vec4 result)
{
  /* TODO(fclem) Finalize with all premult variant. */
  result = color1 * (1.0 - color2.a) + color2;
  result = mix(color1, result, fac);
}
