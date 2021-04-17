
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)

uint bit_field_mask(uint bit_width, uint bit_min)
{
  /* Cannot bit shift more than 31 positions. */
  uint mask = (bit_width > 31u) ? 0x0u : (0xFFFFFFFFu << bit_width);
  return ~mask << bit_min;
}

uint zbin_mask(int word_index, int zbin_min, int zbin_max)
{
  int local_min = clamp(zbin_min - word_index * 32, 0, 31);
  int mask_width = clamp(zbin_max - zbin_min + 1, 0, 32);
  return bit_field_mask(uint(mask_width), uint(local_min));
}

/* Waiting to implement extensions support. We need:
 * - GL_KHR_shader_subgroup_ballot
 * - GL_KHR_shader_subgroup_arithmetic
 * or
 * - Vulkan 1.1
 */
#if 1
#  define subgroupMin(a) a
#  define subgroupMax(a) a
#  define subgroupOr(a) a
#  define subgroupBroadcastFirst(a) a
#endif

#define ITEM_FOREACH_BEGIN(_culling, _tiles_tx, _linearz, _item_index) \
  { \
    int zbin_index = culling_z_to_zbin(_culling, _linearz); \
    zbin_index = min(max(zbin_index, 0), int(CULLING_ZBIN_COUNT - 1)); \
    uint zbin_data = _culling.zbins[zbin_index / 4][zbin_index % 4]; \
    int min_index = int(zbin_data & uint(CULLING_ITEM_BATCH - 1)); \
    int max_index = int((zbin_data >> 16u) & uint(CULLING_ITEM_BATCH - 1)); \
    /* Ensure all threads inside a subgroup get the same value to reduce VGPR usage. */ \
    min_index = subgroupBroadcastFirst(subgroupMin(min_index)); \
    max_index = subgroupBroadcastFirst(subgroupMax(max_index)); \
    int word_min = 0; \
    int word_max = max(0, CULLING_MAX_WORD - 1); \
    word_min = max(min_index / 32, word_min); \
    word_max = min(max_index / 32, word_max); \
    for (int word_index = word_min; word_index <= word_max; word_index++) { \
      /* TODO(fclem) Support bigger max_word with larger texture. */ \
      ivec2 texel = ivec2(gl_FragCoord.xy) / _culling.tile_size; \
      uint word = texelFetch(_tiles_tx, texel, 0)[word_index]; \
      uint mask = zbin_mask(word_index, min_index, max_index); \
      word &= mask; \
      /* Ensure all threads inside a subgroup get the same value to reduce VGPR usage. */ \
      word = subgroupBroadcastFirst(subgroupOr(word)); \
      /* TODO(fclem) Replace by findLSB on supported hardware. */ \
      for (uint i = 0u; word != 0u; word = word >> 1u, i++) { \
        if ((word & 1u) != 0u) { \
          int _item_index = word_index * 32 + int(i);

/* No culling. Iterate over all items. */
#define ITEM_FOREACH_BEGIN_NO_CULL(_culling, _item_index) \
  { \
    { \
      { \
        for (uint _item_index = 0u; _item_index < _culling.items_count; _item_index++) {

#define ITEM_FOREACH_END \
  } \
  } \
  } \
  }
