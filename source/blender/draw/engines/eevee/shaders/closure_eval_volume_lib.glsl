
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
Closure closure_eval(ClosureDiffuse diffuse, ClosureReflection reflection)
{
  return CLOSURE_DEFAULT;
}
Closure closure_eval(ClosureDiffuse diffuse,
                     ClosureReflection reflection,
                     ClosureReflection clearcoat)
{
  return CLOSURE_DEFAULT;
}
Closure closure_eval(ClosureDiffuse diffuse,
                     ClosureReflection reflection,
                     ClosureReflection clearcoat,
                     ClosureRefraction refraction)
{
  return CLOSURE_DEFAULT;
}
Closure closure_eval(ClosureReflection reflection, ClosureReflection clearcoat)
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

vec4 closure_to_rgba(Closure closure)
{
  /* Not supported */
  return vec4(0.0);
}

Closure closure_mix(Closure cl1, Closure cl2, float fac)
{
  Closure cl;
  cl.absorption = mix(cl1.absorption, cl2.absorption, fac);
  cl.scatter = mix(cl1.scatter, cl2.scatter, fac);
  cl.emission = mix(cl1.emission, cl2.emission, fac);
  cl.anisotropy = mix(cl1.anisotropy, cl2.anisotropy, fac);
  return cl;
}

Closure closure_add(Closure cl1, Closure cl2)
{
  Closure cl;
  cl.absorption = cl1.absorption + cl2.absorption;
  cl.scatter = cl1.scatter + cl2.scatter;
  cl.emission = cl1.emission + cl2.emission;
  cl.anisotropy = (cl1.anisotropy + cl2.anisotropy) / 2.0; /* Average phase (no multi lobe) */
  return cl;
}
