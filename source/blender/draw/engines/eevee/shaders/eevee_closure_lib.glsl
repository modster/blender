
struct ClosureDiffuse {
  vec3 color;
  vec3 N;
  float thickness;
  vec3 sss_radius;
  uint sss_id;
};

struct ClosureReflection {
  vec3 color;
  vec3 N;
  float roughness;
};

struct ClosureRefraction {
  vec3 color;
  vec3 N;
  float roughness;
  float ior;
};

struct ClosureVolume {
  vec3 emission;
  vec3 scattering;
  vec3 transmittance;
  float anisotropy;
};

struct ClosureEmission {
  vec3 emission;
};

struct ClosureTransparency {
  vec3 transmittance;
  float holdout;
};

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
  vec4 ssr_data;
  vec2 ssr_normal;
  int flag;
#  ifdef USE_SSS
  vec3 sss_irradiance;
  vec3 sss_albedo;
  float sss_radius;
#  endif

#endif
};

/* clang-format off */
/* Avoid multi-line defines. */
#ifdef VOLUMETRICS
#  define CLOSURE_DEFAULT Closure(vec3(0), vec3(0), vec3(0), 0.0)
#elif !defined(USE_SSS)
#  define CLOSURE_DEFAULT Closure(vec3(0), vec3(0), 0.0, vec4(0), vec2(0), 0)
#else
#  define CLOSURE_DEFAULT Closure(vec3(0), vec3(0), 0.0, vec4(0), vec2(0), 0, vec3(0), vec3(0), 0.0)
#endif
/* clang-format on */
