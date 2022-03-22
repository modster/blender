
#pragma BLENDER_REQUIRE(closure_eval_diffuse_lib.glsl)
#pragma BLENDER_REQUIRE(closure_eval_glossy_lib.glsl)
#pragma BLENDER_REQUIRE(closure_eval_refraction_lib.glsl)
#pragma BLENDER_REQUIRE(closure_eval_translucent_lib.glsl)

bool do_sss = true;
bool do_ssr = true;

vec3 out_sss_radiance;
vec3 out_sss_color;
float out_sss_radius;

float out_ssr_roughness;
vec3 out_ssr_color;
vec3 out_ssr_N;

bool output_sss(ClosureDiffuse diffuse, ClosureOutputDiffuse diffuse_out)
{
  if (diffuse.sss_id == 0u || !do_sss || !sssToggle || outputSssId == 0) {
    return false;
  }
  out_sss_radiance = diffuse_out.radiance;
  out_sss_color = diffuse.color;
  out_sss_radius = avg(diffuse.sss_radius);
  do_sss = false;
  return true;
}

bool output_ssr(ClosureReflection reflection)
{
  if (!do_ssr || !ssrToggle || outputSsrId == 0) {
    return false;
  }
  out_ssr_roughness = reflection.roughness;
  out_ssr_color = reflection.color;
  out_ssr_N = reflection.N;
  do_ssr = false;
  return true;
}

/* Single BSDFs. */
CLOSURE_EVAL_FUNCTION_DECLARE_1(DiffuseBSDF, Diffuse);
Closure closure_inline_eval(ClosureDiffuse diffuse)
{
  /* Glue with the old sytem. */
  CLOSURE_VARS_DECLARE_1(Diffuse);

  in_Diffuse_0.N = diffuse.N;
  in_Diffuse_0.albedo = diffuse.color;

  CLOSURE_EVAL_FUNCTION_1(DiffuseBSDF, Diffuse);

  Closure closure = CLOSURE_DEFAULT;
  if (!output_sss(diffuse, out_Diffuse_0)) {
    closure.radiance += out_Diffuse_0.radiance * diffuse.color;
  }
  return closure;
}

CLOSURE_EVAL_FUNCTION_DECLARE_1(GlossyBSDF, Glossy);
Closure closure_inline_eval(ClosureReflection reflection)
{
  /* Glue with the old sytem. */
  CLOSURE_VARS_DECLARE_1(Glossy);

  in_Glossy_0.N = reflection.N;
  in_Glossy_0.roughness = reflection.roughness;

  CLOSURE_EVAL_FUNCTION_1(GlossyBSDF, Glossy);

  Closure closure = CLOSURE_DEFAULT;
  if (!output_ssr(reflection)) {
    closure.radiance += out_Glossy_0.radiance * reflection.color;
  }
  return closure;
}

CLOSURE_EVAL_FUNCTION_DECLARE_1(RefractionBSDF, Refraction);
Closure closure_inline_eval(ClosureRefraction refraction)
{
  /* Glue with the old sytem. */
  CLOSURE_VARS_DECLARE_1(Refraction);

  in_Refraction_0.N = refraction.N;
  in_Refraction_0.roughness = refraction.roughness;
  in_Refraction_0.ior = refraction.ior;

  CLOSURE_EVAL_FUNCTION_1(RefractionBSDF, Refraction);

  Closure closure = CLOSURE_DEFAULT;
  closure.radiance += out_Refraction_0.radiance * refraction.color;
  return closure;
}

Closure closure_inline_eval(ClosureEmission emission)
{
  Closure closure = CLOSURE_DEFAULT;
  closure.radiance += emission.emission;
  return closure;
}

Closure closure_inline_eval(ClosureTransparency transparency)
{
  Closure closure = CLOSURE_DEFAULT;
  closure.transmittance += transparency.transmittance;
  closure.holdout += transparency.holdout;
  return closure;
}

CLOSURE_EVAL_FUNCTION_DECLARE_2(GlassBSDF, Glossy, Refraction);
Closure closure_inline_eval(ClosureReflection reflection, ClosureRefraction refraction)
{
  /* Glue with the old sytem. */
  CLOSURE_VARS_DECLARE_2(Glossy, Refraction);

  in_Glossy_0.N = reflection.N;
  in_Glossy_0.roughness = reflection.roughness;
  in_Refraction_1.N = refraction.N;
  in_Refraction_1.roughness = refraction.roughness;
  in_Refraction_1.ior = refraction.ior;

  CLOSURE_EVAL_FUNCTION_2(GlassBSDF, Glossy, Refraction);

  Closure closure = CLOSURE_DEFAULT;
  closure.radiance += out_Refraction_1.radiance * refraction.color;
  if (!output_ssr(reflection)) {
    closure.radiance += out_Glossy_0.radiance * reflection.color;
  }
  return closure;
}

CLOSURE_EVAL_FUNCTION_DECLARE_2(SpecularBSDF, Diffuse, Glossy);
Closure closure_inline_eval(ClosureDiffuse diffuse,
                            ClosureReflection reflection,
                            ClosureEmission emission,
                            ClosureTransparency transparency)
{
  /* Glue with the old sytem. */
  CLOSURE_VARS_DECLARE_2(Diffuse, Glossy);

  in_Diffuse_0.N = diffuse.N;
  in_Diffuse_0.albedo = diffuse.color;
  in_Glossy_1.N = reflection.N;
  in_Glossy_1.roughness = reflection.roughness;

  CLOSURE_EVAL_FUNCTION_2(SpecularBSDF, Diffuse, Glossy);

  Closure closure = CLOSURE_DEFAULT;
  closure.radiance += out_Diffuse_0.radiance * diffuse.color;
  if (!output_ssr(reflection)) {
    closure.radiance += out_Glossy_1.radiance * reflection.color;
  }
  closure.radiance += emission.emission;
  closure.transmittance = transparency.transmittance;
  closure.holdout = transparency.holdout;
  return closure;
}

CLOSURE_EVAL_FUNCTION_DECLARE_4(PrincipledBSDF, Diffuse, Glossy, Glossy, Refraction);
Closure closure_inline_eval(ClosureDiffuse diffuse,
                            ClosureReflection reflection,
                            ClosureReflection clearcoat,
                            ClosureRefraction refraction,
                            ClosureEmission emission,
                            ClosureTransparency transparency)
{
  /* Glue with the old sytem. */
  CLOSURE_VARS_DECLARE_4(Diffuse, Glossy, Glossy, Refraction);

  in_Diffuse_0.N = diffuse.N;
  in_Diffuse_0.albedo = diffuse.color;
  in_Glossy_1.N = reflection.N;
  in_Glossy_1.roughness = reflection.roughness;
  in_Glossy_2.N = clearcoat.N;
  in_Glossy_2.roughness = clearcoat.roughness;
  in_Refraction_3.N = refraction.N;
  in_Refraction_3.roughness = refraction.roughness;
  in_Refraction_3.ior = refraction.ior;

  CLOSURE_EVAL_FUNCTION_4(PrincipledBSDF, Diffuse, Glossy, Glossy, Refraction);

  Closure closure = CLOSURE_DEFAULT;
  closure.radiance += out_Glossy_2.radiance * clearcoat.color +
                      out_Refraction_3.radiance * refraction.color;
  if (!output_sss(diffuse, out_Diffuse_0)) {
    closure.radiance += out_Diffuse_0.radiance * diffuse.color;
  }
  if (!output_ssr(reflection)) {
    closure.radiance += out_Glossy_1.radiance * reflection.color;
  }
  closure.radiance += emission.emission;
  closure.transmittance = transparency.transmittance;
  closure.holdout = transparency.holdout;
  return closure;
}
