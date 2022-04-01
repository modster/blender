
bool do_sss = true;
bool do_ssr = true;

vec3 out_sss_radiance;
vec3 out_sss_color;
float out_sss_radius;

float out_ssr_roughness;
vec3 out_ssr_color;
vec3 out_ssr_N;

/* Surface BSDFs. */
Closure closure_eval(ClosureDiffuse diffuse)
{
  return CLOSURE_DEFAULT;
}
Closure closure_eval(ClosureReflection reflection)
{
  return CLOSURE_DEFAULT;
}
Closure closure_eval(ClosureRefraction refraction)
{
  return CLOSURE_DEFAULT;
}
Closure closure_eval(ClosureEmission emission)
{
  return CLOSURE_DEFAULT;
}
Closure closure_eval(ClosureTransparency transparency)
{
  return CLOSURE_DEFAULT;
}
Closure closure_eval(ClosureReflection reflection, ClosureRefraction refraction)
{
  return CLOSURE_DEFAULT;
}
Closure closure_eval(ClosureDiffuse diffuse,
                     ClosureReflection reflection,
                     ClosureEmission emission,
                     ClosureTransparency transparency)
{
  return CLOSURE_DEFAULT;
}
Closure closure_eval(ClosureDiffuse diffuse,
                     ClosureReflection reflection,
                     ClosureReflection clearcoat,
                     ClosureRefraction refraction,
                     ClosureEmission emission,
                     ClosureTransparency transparency)
{
  return CLOSURE_DEFAULT;
}

/* TODO(fclem) Implement. */
Closure closure_eval(ClosureVolumeScatter volume_scatter)
{
  return CLOSURE_DEFAULT;
}
Closure closure_eval(ClosureVolumeAbsorption volume_absorption)
{
  return CLOSURE_DEFAULT;
}
Closure closure_eval(ClosureVolumeScatter volume_scatter,
                     ClosureVolumeAbsorption volume_absorption,
                     ClosureEmission emission)
{
  return CLOSURE_DEFAULT;
}
