void node_bsdf_toon(vec4 color, float size, float tsmooth, vec3 N,float weight, out Closure result)
{
  node_bsdf_diffuse(color, 0.0, N, result);
}
