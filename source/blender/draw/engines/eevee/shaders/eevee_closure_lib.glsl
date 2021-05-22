
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

/* Store weight in red channel. Store negative to differentiate with evaluated closure. */
void closure_weight_add(inout ClosureDiffuse closure, float weight)
{
  closure.color.r -= weight;
}
void closure_weight_add(inout ClosureReflection closure, float weight)
{
  closure.color.r -= weight;
}
void closure_weight_add(inout ClosureRefraction closure, float weight)
{
  closure.color.r -= weight;
}

/* Create a random threshold inside the weight range. */
void closure_weight_randomize(inout ClosureDiffuse closure, float randu)
{
  closure.color.g = closure.color.r * randu;
}
void closure_weight_randomize(inout ClosureReflection closure, float randu)
{
  closure.color.g = closure.color.r * randu;
}
void closure_weight_randomize(inout ClosureRefraction closure, float randu)
{
  closure.color.g = closure.color.r * randu;
}

bool closure_weight_threshold(inout ClosureDiffuse closure, inout float weight)
{
  /* Check if closure has not yet been evaluated. */
  if (closure.color.r < 0.0) {
    /* Decrement weight from random threshold. */
    closure.color.g += weight;
    /* Evaluate this closure if threshold reaches 0. */
    if (closure.color.g >= 0.0) {
      /* Returns the sum of all weights. */
      weight = abs(closure.color.r);
      return true;
    }
  }
  return false;
}
bool closure_weight_threshold(inout ClosureReflection closure, inout float weight)
{
  if (closure.color.r < 0.0) {
    closure.color.g += weight;
    if (closure.color.g >= 0.0) {
      weight = abs(closure.color.r);
      return true;
    }
  }
  return false;
}
bool closure_weight_threshold(inout ClosureRefraction closure, inout float weight)
{
  if (closure.color.r < 0.0) {
    closure.color.g += weight;
    if (closure.color.g >= 0.0) {
      weight = abs(closure.color.r);
      return true;
    }
  }
  return false;
}
