void node_background(vec4 color, float strength, out Closure result)
{
  g_emission_data.emission = color.rgb * strength;
}
