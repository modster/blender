void node_bsdf_transparent(vec4 color, out Closure result)
{
  g_transparency_data.transmittance = color.rgb;
}
