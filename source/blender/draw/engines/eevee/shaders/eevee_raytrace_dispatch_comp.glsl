/**
 * Fix dispatch arguments to fit implementation limits.
 */

#pragma BLENDER_REQUIRE(common_math_lib.glsl)

/* Minimum limit garanteed. */
const uint dispatch_limit = 0xFFu;

uint low_remainder_divisor_search(uint dividend)
{
  uint best_divisor = divide_ceil_u(dividend, dispatch_limit);
  uint best_remainder = best_divisor * dispatch_limit - dividend;
  /* Really dumb for now. */
  /* Search for divisor up to 1024 because 1024² is the max amount of tiles for a 16K render. */
  for (uint divisor = best_divisor + 1; divisor <= 1024 && best_remainder != 0u; divisor++) {
    uint remainder = divide_ceil_u(dividend, divisor) * divisor - dividend;
    if (remainder < best_remainder) {
      best_remainder = remainder;
      best_divisor = divisor;
    }
  }
  return best_divisor;
}

void main()
{
  uint tile_count = dispatch_buf.num_groups_x;
  if (tile_count > dispatch_limit) {
    uint divisor = low_remainder_divisor_search(tile_count);
    dispatch_buf.num_groups_x = divide_ceil_u(tile_count, divisor);
    dispatch_buf.num_groups_y = divisor;
    dispatch_buf.tile_count = tile_count;
  }
}
