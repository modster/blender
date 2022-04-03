
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
vec4 closure_to_rgba(Closure);

/* Single BSDFs. */
Closure closure_eval(ClosureDiffuse diffuse);
Closure closure_eval(ClosureReflection reflection);
Closure closure_eval(ClosureRefraction refraction);
Closure closure_eval(ClosureEmission emission);
Closure closure_eval(ClosureTransparency transparency);
Closure closure_eval(ClosureVolumeScatter volume_scatter);
Closure closure_eval(ClosureVolumeAbsorption volume_absorption);
/* Glass BSDF. */
Closure closure_eval(ClosureReflection reflection, ClosureRefraction refraction);
/* Dielectric BSDF. */
Closure closure_eval(ClosureDiffuse diffuse, ClosureReflection reflection);
/* ClearCoat BSDF. */
Closure closure_eval(ClosureReflection reflection, ClosureReflection clearcoat);
/* Volume BSDF. */
Closure closure_eval(ClosureVolumeScatter volume_scatter,
                     ClosureVolumeAbsorption volume_absorption,
                     ClosureEmission emission);
/* Specular BSDF. */
Closure closure_eval(ClosureDiffuse diffuse,
                     ClosureReflection reflection,
                     ClosureReflection clearcoat);
/* Principled BSDF. */
Closure closure_eval(ClosureDiffuse diffuse,
                     ClosureReflection reflection,
                     ClosureReflection clearcoat,
                     ClosureRefraction refraction);

Closure closure_add(Closure cl1, Closure cl2);
Closure closure_mix(Closure cl1, Closure cl2, float fac);

float ambient_occlusion_eval(vec3 normal,
                             float distance,
                             const float inverted,
                             const float sample_count);

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

#ifdef VOLUMETRICS
#  define CLOSURE_DEFAULT Closure(vec3(0), vec3(0), vec3(0), 0.0)
#else
#  define CLOSURE_DEFAULT Closure(vec3(0), vec3(0), 0.0)
#endif
