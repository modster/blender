
/**
 * Samples a 3x3 tile neighborhood to find potentially intersecting motions.
 * Outputs the largest intersecting motion vector in the neighboorhod.
 *
 * Based on:
 * A Fast and Stable Feature-Aware Motion Blur Filter
 * by Jean-Philippe Guertin, Morgan McGuire, Derek Nowrouzezahrai
 *
 * Adapted from G3D Innovation Engine implementation.
 */

#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)

layout(std140) uniform motion_blur_block
{
  MotionBlurData mb;
};

uniform sampler2D tiles_tx;

layout(location = 0) out vec4 out_max_motion;

bool neighbor_affect_this_tile(ivec2 offset, vec2 velocity)
{
  /* Manhattan distance to the tiles, which is used for
   * differentiating corners versus middle blocks */
  float displacement = float(abs(offset.x) + abs(offset.y));
  /**
   * Relative sign on each axis of the offset compared
   * to the velocity for that tile.  In order for a tile
   * to affect the center tile, it must have a
   * neighborhood velocity in which x and y both have
   * identical or both have opposite signs relative to
   * offset. If the offset coordinate is zero then
   * velocity is irrelevant.
   */
  vec2 point = sign(offset * velocity);

  float dist = (point.x + point.y);
  /**
   * Here's an example of the logic for this code.
   * In this diagram, the upper-left tile has offset = (-1, -1).
   * V1 is velocity = (1, -2). point in this case = (-1, 1), and therefore dist = 0,
   * so the upper-left tile does not affect the center.
   *
   * Now, look at another case. V2 = (-1, -2). point = (1, 1), so dist = 2 and the tile
   * does affect the center.
   *
   * V2(-1,-2)  V1(1, -2)
   *        \    /
   *         \  /
   *          \/___ ____ ____
   *  (-1, -1)|    |    |    |
   *          |____|____|____|
   *          |    |    |    |
   *          |____|____|____|
   *          |    |    |    |
   *          |____|____|____|
   */
  return (abs(dist) == displacement);
}

/**
 * Only gather neighborhood velocity into tiles that could be affected by it.
 * In the general case, only six of the eight neighbors contribute:
 *
 *  This tile can't possibly be affected by the center one
 *     |
 *     v
 *    ____ ____ ____
 *   |    | ///|/// |
 *   |____|////|//__|
 *   |    |////|/   |
 *   |___/|////|____|
 *   |  //|////|    | <--- This tile can't possibly be affected by the center one
 *   |_///|///_|____|
 */
void main()
{
  ivec2 tile = ivec2(gl_FragCoord.xy);
  ivec2 texture_bounds = textureSize(tiles_tx, 0) - 1;

  out_max_motion = vec4(0.0);
  float max_motion_len_sqr_prev = -1.0;
  float max_motion_len_sqr_next = -1.0;

  ivec2 offset = ivec2(0);
  for (offset.y = -1; offset.y <= 1; ++offset.y) {
    for (offset.x = -1; offset.x <= 1; ++offset.x) {
      ivec2 sample_tile = clamp(tile + offset, ivec2(0), texture_bounds);
      vec4 motion = texelFetch(tiles_tx, sample_tile, 0);

      float motion_len_sqr_prev = len_squared(motion.xy);
      float motion_len_sqr_next = len_squared(motion.zw);

      if (motion_len_sqr_prev > max_motion_len_sqr_prev) {
        if (neighbor_affect_this_tile(offset, motion.xy)) {
          max_motion_len_sqr_prev = motion_len_sqr_prev;
          out_max_motion.xy = motion.xy;
        }
      }

      if (motion_len_sqr_next > max_motion_len_sqr_next) {
        if (neighbor_affect_this_tile(offset, motion.zw)) {
          max_motion_len_sqr_next = motion_len_sqr_next;
          out_max_motion.zw = motion.zw;
        }
      }
    }
  }
}
