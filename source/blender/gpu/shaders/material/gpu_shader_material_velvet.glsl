void node_bsdf_velvet(vec4 color, float sigma, vec3 N, float weight, out Closure result)
{
  node_bsdf_diffuse(color, 0.0, N, weight, result);
}
