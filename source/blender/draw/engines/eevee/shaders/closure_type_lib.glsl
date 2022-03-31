
#pragma BLENDER_REQUIRE(gpu_shader_codegen_lib.glsl)
/* #pragma (common_math_geom_lib.glsl) */
/* #pragma (common_uniforms_lib.glsl) */
/* #pragma (renderpass_lib.glsl) */

#ifndef VOLUMETRICS

uniform int outputSsrId = 1;
uniform int outputSssId = 1;

#endif

struct Closure {
#ifdef VOLUMETRICS
  vec3 absorption;
  vec3 scatter;
  vec3 emission;
  float anisotropy;

#else /* SURFACE */
  vec3 radiance;
  vec3 transmittance;
  float holdout;
#endif
};

#ifndef GPU_METAL
/* Prototype */
Closure nodetree_exec();
/* Single BSDFs. */
Closure closure_inline_eval(ClosureDiffuse diffuse);
Closure closure_inline_eval(ClosureReflection reflection);
Closure closure_inline_eval(ClosureRefraction refraction);
Closure closure_inline_eval(ClosureEmission emission);
Closure closure_inline_eval(ClosureTransparency transparency);
/* Glass BSDF. */
Closure closure_inline_eval(ClosureReflection reflection, ClosureRefraction refraction);
/* Specular BSDF. */
Closure closure_inline_eval(ClosureDiffuse diffuse,
                            ClosureReflection reflection,
                            ClosureEmission emission,
                            ClosureTransparency transparency);
/* Principled BSDF. */
Closure closure_inline_eval(ClosureDiffuse diffuse,
                            ClosureReflection reflection,
                            ClosureReflection clearcoat,
                            ClosureRefraction refraction,
                            ClosureEmission emission,
                            ClosureTransparency transparency);
/* WORKAROUND: Included later with libs. This is because we are mixing include systems. */
vec3 safe_normalize(vec3 N);
float fast_sqrt(float a);
vec3 cameraVec(vec3 P);
vec2 btdf_lut(float a, float b, float c);
vec2 brdf_lut(float a, float b);
vec3 F_brdf_multi_scatter(vec3 a, vec3 b, vec2 c);
vec3 F_brdf_single_scatter(vec3 a, vec3 b, vec2 c);
float F_eta(float a, float b);
#endif

/* Not used */
#define closure_weight_threshold(A, B) true
#define ntree_eval_init()
#define ntree_eval_weights()

#ifdef VOLUMETRICS
#  define CLOSURE_DEFAULT Closure(vec3(0), vec3(0), vec3(0), 0.0)
#else
#  define CLOSURE_DEFAULT Closure(vec3(0), vec3(0), 0.0)
#endif

#ifdef VOLUMETRICS
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

#else /* SURFACE */

Closure closure_add(Closure cl1, Closure cl2)
{
  Closure cl;
  cl.radiance = cl1.radiance + cl2.radiance;
  cl.transmittance = cl1.transmittance + cl2.transmittance;
  cl.holdout = cl1.holdout + cl2.holdout;
  return cl;
}

Closure closure_mix(Closure cl1, Closure cl2, float fac)
{
  /* Weights have already been applied. */
  return closure_add(cl1, cl2);
}

#endif
