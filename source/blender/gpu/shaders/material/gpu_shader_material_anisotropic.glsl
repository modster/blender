
void node_bsdf_anisotropic(vec4 color,
                           float roughness,
                           float anisotropy,
                           vec3 N,
                           vec3 T,
                           float weight,
                           float use_multiscatter,
                           out Closure result)
{
  ClosureReflection reflection_data;
  reflection_data.weight = weight;
  /* TODO(fclem): Multiscatter. */
  reflection_data.color = color.rgb;
  reflection_data.N = N;
  reflection_data.roughness = roughness;

  result = closure_eval(reflection_data);
}
