
/**
 * Perform two gather blur in the 2 motion blur directions
 * Based on:
 * A Fast and Stable Feature-Aware Motion Blur Filter
 * by Jean-Philippe Guertin, Morgan McGuire, Derek Nowrouzezahrai
 *
 * With modification from the presentation:
 * Next Generation Post Processing in Call of Duty Advanced Warfare
 * by Jorge Jimenez
 */

#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_motion_blur_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_sampling_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)

layout(std140) uniform sampling_block
{
  SamplingData sampling;
};

layout(std140) uniform motion_blur_block
{
  MotionBlurData mb;
};

uniform sampler2D color_tx;
uniform sampler2D depth_tx;
uniform sampler2D velocity_tx;
uniform sampler2D tiles_tx;

in vec4 uvcoordsvar;

layout(location = 0) out vec4 out_color;

const int gather_sample_count = 8;

vec2 spread_compare(float center_motion_length, float sample_motion_length, float offset_length)
{
  return saturate(vec2(center_motion_length, sample_motion_length) - offset_length + 1.0);
}

vec2 depth_compare(float center_depth, float sample_depth)
{
  return saturate(0.5 + vec2(-mb.depth_scale, mb.depth_scale) * (sample_depth - center_depth));
}

/* Kill contribution if not going the same direction. */
float dir_compare(vec2 offset, vec2 sample_motion, float sample_motion_length)
{
  if (sample_motion_length < 0.5) {
    return 1.0;
  }
  return (dot(offset, sample_motion) > 0.0) ? 1.0 : 0.0;
}

/* Return background (x) and foreground (y) weights. */
vec2 sample_weights(float center_depth,
                    float sample_depth,
                    float center_motion_length,
                    float sample_motion_length,
                    float offset_length)
{
  /* Classify foreground/background. */
  vec2 depth_weight = depth_compare(center_depth, sample_depth);
  /* Weight if sample is overlapping or under the center pixel. */
  vec2 spread_weight = spread_compare(center_motion_length, sample_motion_length, offset_length);
  return depth_weight * spread_weight;
}

void gather_sample(vec2 screen_uv,
                   float center_depth,
                   float center_motion_len,
                   vec2 offset,
                   float offset_len,
                   const bool next,
                   inout vec4 accum,
                   inout vec4 accum_bg,
                   inout vec3 w_accum)
{
  vec2 sample_uv = screen_uv - offset * mb.target_size_inv;
  vec2 sample_motion = sample_velocity(mb, velocity_tx, sample_uv, next);
  float sample_motion_len = length(sample_motion);
  float sample_depth = texture(depth_tx, sample_uv).r;
  vec4 sample_color = textureLod(color_tx, sample_uv, 0.0);

  /* Meh, a quirk of the motion vector pass... */
  sample_motion = (next) ? -sample_motion : sample_motion;

  sample_depth = get_view_z_from_depth(sample_depth);

  vec3 weights;
  weights.xy = sample_weights(
      center_depth, sample_depth, center_motion_len, sample_motion_len, offset_len);
  weights.z = dir_compare(offset, sample_motion, sample_motion_len);
  weights.xy *= weights.z;

  accum += sample_color * weights.y;
  accum_bg += sample_color * weights.x;
  w_accum += weights;
}

void gather_blur(vec2 screen_uv,
                 vec2 center_motion,
                 float center_depth,
                 vec2 max_motion,
                 float ofs,
                 const bool next,
                 inout vec4 accum,
                 inout vec4 accum_bg,
                 inout vec3 w_accum)
{
  float center_motion_len = length(center_motion);
  float max_motion_len = length(max_motion);

  /* Tile boundaries randomization can fetch a tile where there is less motion than this pixel.
   * Fix this by overriding the max_motion. */
  if (max_motion_len < center_motion_len) {
    max_motion_len = center_motion_len;
    max_motion = center_motion;
  }

  if (max_motion_len < 0.5) {
    return;
  }

  int i;
  float t, inc = 1.0 / float(gather_sample_count);
  for (i = 0, t = ofs * inc; i < gather_sample_count; i++, t += inc) {
    gather_sample(screen_uv,
                  center_depth,
                  center_motion_len,
                  max_motion * t,
                  max_motion_len * t,
                  next,
                  accum,
                  accum_bg,
                  w_accum);
  }

  if (center_motion_len < 0.5) {
    return;
  }

  for (i = 0, t = ofs * inc; i < gather_sample_count; i++, t += inc) {
    /* Also sample in center motion direction.
     * Allow recovering motion where there is conflicting
     * motion between foreground and background. */
    gather_sample(screen_uv,
                  center_depth,
                  center_motion_len,
                  center_motion * t,
                  center_motion_len * t,
                  next,
                  accum,
                  accum_bg,
                  w_accum);
  }
}

void main()
{
  vec2 uv = uvcoordsvar.xy;

  /* Data of the center pixel of the gather (target). */
  float center_depth = get_view_z_from_depth(texture(depth_tx, uv).r);
  vec4 center_motion = sample_velocity(mb, velocity_tx, ivec2(gl_FragCoord.xy));

  vec4 center_color = textureLod(color_tx, uv, 0.0);

  float noise_offset = sampling_rng_1D_get(sampling, SAMPLING_TIME);
  /** TODO(fclem) Blue noise. */
  vec2 rand = vec2(interlieved_gradient_noise(gl_FragCoord.xy, 0, noise_offset),
                   interlieved_gradient_noise(gl_FragCoord.xy, 1, noise_offset));

  /* Randomize tile boundary to avoid ugly discontinuities. Randomize 1/4th of the tile.
   * Note this randomize only in one direction but in practice it's enough. */
  rand.x = rand.x * 2.0 - 1.0;
  ivec2 tile = ivec2(gl_FragCoord.xy + rand.x * float(MB_TILE_DIVISOR) * 0.25) / MB_TILE_DIVISOR;
  tile = clamp(tile, ivec2(0), textureSize(tiles_tx, 0) - 1);
  vec4 max_motion = texelFetch(tiles_tx, tile, 0);

  /* First (center) sample: time = T */
  /* x: Background, y: Foreground, z: dir. */
  vec3 w_accum = vec3(0.0, 0.0, 1.0);
  vec4 accum_bg = vec4(0.0);
  vec4 accum = vec4(0.0);
  /* First linear gather. time = [T - delta, T] */
  gather_blur(
      uv, center_motion.xy, center_depth, max_motion.xy, rand.y, false, accum, accum_bg, w_accum);
  /* Second linear gather. time = [T, T + delta] */
  gather_blur(
      uv, -center_motion.zw, center_depth, -max_motion.zw, rand.y, true, accum, accum_bg, w_accum);

#if 1 /* Own addition. Not present in reference implementation. */
  /* Avoid division by 0.0. */
  float w = 1.0 / (50.0 * float(gather_sample_count) * 4.0);
  accum_bg += center_color * w;
  w_accum.x += w;
  /* NOTE: In Jimenez's presentation, they used center sample.
   * We use background color as it contains more information for foreground
   * elements that have not enough weights.
   * Yield better blur in complex motion. */
  center_color = accum_bg / w_accum.x;
#endif
  /* Merge background. */
  accum += accum_bg;
  w_accum.y += w_accum.x;
  /* Balance accumulation for failed samples.
   * We replace the missing foreground by the background. */
  float blend_fac = saturate(1.0 - w_accum.y / w_accum.z);
  out_color = (accum / w_accum.z) + center_color * blend_fac;

#if 0 /* For debugging. */
  out_color.rgb = out_color.ggg;
  out_color.rg += max_motion.xy;
#endif
}
