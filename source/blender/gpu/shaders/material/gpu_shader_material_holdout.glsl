
void node_holdout(float weight, out Closure result)
{
  ClosureTransparency transparency_data;
  transparency_data.transmittance = vec3(0.0);
  transparency_data.holdout = weight;

  result = closure_eval(transparency_data);
}
