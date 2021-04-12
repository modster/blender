
/**
 * Shaders that down-sample velocity buffer into squared tile of MB_TILE_DIVISOR pixels wide.
 * Outputs the largest motion vector in the tile area.
 *
 * Based on:
 * A Fast and Stable Feature-Aware Motion Blur Filter
 * by Jean-Philippe Guertin, Morgan McGuire, Derek Nowrouzezahrai
 *
 * Adapted from G3D Innovation Engine implementation.
 */

#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_motion_blur_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)

layout(std140) uniform motion_blur_block
{
  MotionBlurData mb;
};

uniform sampler2D velocity_tx;

layout(location = 0) out vec4 out_max_motion;

void main()
{
  ivec2 texture_bounds = textureSize(velocity_tx, 0) - 1;
  ivec2 tile_co = ivec2(gl_FragCoord.xy);

  float max_motion_len_sqr_prev = -1.0;
  float max_motion_len_sqr_next = -1.0;
  for (int x = 0; x < MB_TILE_DIVISOR; x++) {
    for (int y = 0; y < MB_TILE_DIVISOR; y++) {
      ivec2 sample_texel = tile_co * MB_TILE_DIVISOR + ivec2(x, y);
      vec4 motion = sample_velocity(mb, velocity_tx, min(sample_texel, texture_bounds));

      float motion_len_sqr_prev = len_squared(motion.xy);
      float motion_len_sqr_next = len_squared(motion.zw);

      if (motion_len_sqr_prev > max_motion_len_sqr_prev) {
        max_motion_len_sqr_prev = motion_len_sqr_prev;
        out_max_motion.xy = motion.xy;
      }
      if (motion_len_sqr_next > max_motion_len_sqr_next) {
        max_motion_len_sqr_next = motion_len_sqr_next;
        out_max_motion.zw = motion.zw;
      }
    }
  }
}
