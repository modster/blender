void node_bsdf_anisotropic(vec4 color,
                           float roughness,
                           float anisotropy,
                           float rotation,
                           vec3 N,
                           vec3 T,
                           const float use_multiscatter,
                           out Closure result)
{
  node_bsdf_glossy(color, roughness, N, use_multiscatter, result);
}
