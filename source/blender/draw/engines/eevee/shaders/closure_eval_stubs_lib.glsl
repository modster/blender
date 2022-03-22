
bool do_sss = true;
bool do_ssr = true;

vec3 out_sss_radiance;
vec3 out_sss_color;
float out_sss_radius;

float out_ssr_roughness;
vec3 out_ssr_color;
vec3 out_ssr_N;

/* Single BSDFs. */
Closure closure_inline_eval(ClosureDiffuse diffuse)
{
  return CLOSURE_DEFAULT;
}

Closure closure_inline_eval(ClosureReflection reflection)
{
  return CLOSURE_DEFAULT;
}

Closure closure_inline_eval(ClosureRefraction refraction)
{
  return CLOSURE_DEFAULT;
}

Closure closure_inline_eval(ClosureEmission emission)
{
  return CLOSURE_DEFAULT;
}

Closure closure_inline_eval(ClosureTransparency transparency)
{
  return CLOSURE_DEFAULT;
}

Closure closure_inline_eval(ClosureReflection reflection, ClosureRefraction refraction)
{
  return CLOSURE_DEFAULT;
}

Closure closure_inline_eval(ClosureDiffuse diffuse,
                            ClosureReflection reflection,
                            ClosureEmission emission,
                            ClosureTransparency transparency)
{
  return CLOSURE_DEFAULT;
}

Closure closure_inline_eval(ClosureDiffuse diffuse,
                            ClosureReflection reflection,
                            ClosureReflection clearcoat,
                            ClosureRefraction refraction,
                            ClosureEmission emission,
                            ClosureTransparency transparency)
{
  return CLOSURE_DEFAULT;
}
