
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)

uint zbin_mask(uint word_index, uint zbin_min, uint zbin_max)
{
  uint word_start = word_index * 32u;
  uint word_end = word_start + 31u;
  uint local_min = max(zbin_min, word_start);
  uint local_max = min(zbin_max, word_end);
  uint mask_width = local_max - local_min + 1;
  return bit_field_mask(mask_width, local_min);
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

#define ITEM_FOREACH_BEGIN(_culling, _zbins, _words, _pixel, _linearz, _item_index) \
  { \
    uint batch_count = divide_ceil_u(_culling.visible_count, CULLING_BATCH_SIZE); \
    uvec2 tile_co = uvec2(_pixel) / _culling.tile_size; \
    uint tile_word_offset = (tile_co.x + tile_co.y * _culling.tile_x_len) * \
                            _culling.tile_word_len; \
    for (uint batch = 0; batch < batch_count; batch++) { \
      int zbin_index = culling_z_to_zbin(_culling, _linearz); \
      zbin_index = clamp(zbin_index, 0, CULLING_ZBIN_COUNT - 1); \
      uint zbin_data = _zbins[zbin_index + batch * CULLING_ZBIN_COUNT]; \
      uint min_index = zbin_data & 0xFFFFu; \
      uint max_index = zbin_data >> 16u; \
      /* Ensure all threads inside a subgroup get the same value to reduce VGPR usage. */ \
      min_index = subgroupBroadcastFirst(subgroupMin(min_index)); \
      max_index = subgroupBroadcastFirst(subgroupMax(max_index)); \
      uint word_min = min_index / 32u; \
      uint word_max = max_index / 32u; \
      for (uint word_idx = word_min; word_idx <= word_max; word_idx++) { \
        uint word = _words[tile_word_offset + word_idx]; \
        word &= zbin_mask(word_idx, min_index, max_index); \
        /* Ensure all threads inside a subgroup get the same value to reduce VGPR usage. */ \
        word = subgroupBroadcastFirst(subgroupOr(word)); \
        while (word != 0u) { \
          uint bit_index = uint(findLSB(word)); \
          word &= ~1u << bit_index; \
          uint _item_index = word_idx * 32u + bit_index;

/* No culling. Iterate over all items. */
#define ITEM_FOREACH_BEGIN_NO_CULL(_culling, _item_index) \
  { \
    { \
      { \
        for (uint _item_index = 0u; _item_index < _culling.visible_count; _item_index++) {

#define ITEM_FOREACH_END \
  } \
  } \
  } \
  }
