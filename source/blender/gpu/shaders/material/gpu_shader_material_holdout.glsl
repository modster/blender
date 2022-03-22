void node_holdout(float weight, out Closure result)
{
#ifdef GPU_NODES_SAMPLE_BSDF
  g_transparency_data.holdout += weight;
#else
  result = closure_inline_eval(ClosureTransparency(vec3(0.0), weight));
#endif
}
