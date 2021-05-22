void node_bsdf_transparent(vec4 color, float weight, out Closure result)
{
  g_transparency_data.transmittance += color.rgb * weight;
}
