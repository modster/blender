
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
