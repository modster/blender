void node_bsdf_transparent(vec4 color, float weight, out Closure result)
{
#ifdef GPU_NODES_SAMPLE_BSDF
  g_transparency_data.transmittance += color.rgb * weight;
#else
  result = closure_inline_eval(ClosureTransparency(color.rgb * weight, 0.0));
#endif
}
