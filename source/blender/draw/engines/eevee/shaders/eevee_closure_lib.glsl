
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

/* We use the weight tree pre-evaluation to weight the closures.
 * There is no need for the Closure type. */
struct Closure {
  float dummy;
};
#define CLOSURE_DEFAULT Closure(0.0)
#define closure_add(a, b) CLOSURE_DEFAULT
#define closure_mix(a, b, c) CLOSURE_DEFAULT
