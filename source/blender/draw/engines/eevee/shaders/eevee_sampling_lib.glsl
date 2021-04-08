
/**
 * Sampling data accessors and random number generators.
 * Also contains some sample mapping functions.
 **/

#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)

/* -------------------------------------------------------------------- */
/** \name Sampling data.
 *
 * Return a random values from Low Discrepency Sequence in [0..1) range.
 * This value is uniform (constant) for the whole scene sample.
 * You might want to couple it with a noise function.
 * \{ */

float sampling_rng_1D_get(SamplingData data, const eSamplingDimension dimension)
{
  return data.dimensions[dimension].x;
}

vec2 sampling_rng_2D_get(SamplingData data, const eSamplingDimension dimension)
{
  return vec2(data.dimensions[dimension].x, data.dimensions[dimension + 1u].x);
}

vec3 sampling_rng_3D_get(SamplingData data, const eSamplingDimension dimension)
{
  return vec3(data.dimensions[dimension].x,
              data.dimensions[dimension + 1u].x,
              data.dimensions[dimension + 2u].x);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Random Number Generators.
 * \{ */

/* Interlieved gradient noise by Jorge Jimenez
 * http://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare
 * Seeding found by Epic Game. */
float interlieved_gradient_noise(vec2 pixel, float seed, float offset)
{
  pixel += seed * (vec2(47, 17) * 0.695);
  return fract(offset + 52.9829189 * fract(0.06711056 * pixel.x + 0.00583715 * pixel.y));
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Random Number Generators.
 * \{ */

/* Given 2 randome number in [0..1] range, return a random unit disk sample. */
vec2 disk_sample(vec2 noise)
{
  float angle = noise.x * M_2PI;
  return vec2(cos(angle), sin(angle)) * sqrt(noise.y);
}

/** \} */
