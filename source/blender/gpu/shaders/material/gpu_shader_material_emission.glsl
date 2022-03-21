void node_emission(vec4 color, float strength, float weight, out Closure result)
{
  g_emission_data.emission += color.rgb * strength * weight;
}
