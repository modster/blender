

/* Interlieved gradient noise by Jorge Jimenez
 * http://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare
 * Seeding found by Epic Game. */
float interlieved_gradient_noise(vec2 pixel, float seed, float offset)
{
  pixel += seed * (vec2(47, 17) * 0.695);
  return fract(offset + 52.9829189 * fract(0.06711056 * pixel.x + 0.00583715 * pixel.y));
}

/* Given 2 randome number in [0..1] range, return a random unit disk sample. */
vec2 disk_sample(vec2 noise)
{
  float angle = noise.x * M_2PI;
  return vec2(cos(angle), sin(angle)) * sqrt(noise.y);
}