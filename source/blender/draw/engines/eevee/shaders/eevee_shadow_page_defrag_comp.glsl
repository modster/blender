
/**
 * Virtual shadowmapping: Defrag.
 *
 * Defragment the cached page buffer making one continuous array.
 * Also pop_front the cached pages if there is not enough free pages for the needed allocations.
 * Here is an example of the behavior of this buffer during one update cycle:
 *
 *   Initial state: 5 cached pages. Buffer starts at index 2 and ends at 6.
 *     [--xxxxx---------]
 *   After page free step: 2 cached pages were removed (r), 3 pages were inserted in the cache (i).
 *     [--xrxrxiii------]
 *   After page defrag step: The buffer is compressed into only 6 pages.
 *     [----xxxxxx------]
 */

#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shadow_page_ops_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shadow_tilemap_lib.glsl)

void find_first_valid(inout uint src, uint dst)
{
  for (; src < dst; src++) {
    if (pages_cached_buf[src % uint(SHADOW_MAX_PAGE)].x != uint(-1)) {
      return;
    }
  }
}

void page_cached_free(uint index)
{
  ivec2 texel = ivec2(unpackUvec2x16(pages_cached_buf[index].y));
  ShadowTileData tile = shadow_tile_data_unpack(imageLoad(tilemaps_img, texel).x);

  shadow_page_cached_buf_remove(tile);
  shadow_page_free_buf_append(tile);

  imageStore(tilemaps_img, texel, uvec4(shadow_tile_data_pack(tile)));
}

void main()
{
  const uint max_page = SHADOW_MAX_PAGE;
  /* Pages we need to get off the cache for the allocation pass. */
  int additional_pages = pages_infos_buf.page_alloc_count - pages_infos_buf.page_free_count;

  uint src = pages_infos_buf.page_cached_start;
  uint end = pages_infos_buf.page_cached_end;

  find_first_valid(src, end);

  /* First free pages from the defrag range. Avoid defragmenting to then free them. */
  for (; additional_pages > 0 && src < end; additional_pages--) {
    page_cached_free(src % max_page);
    find_first_valid(src, end);
  }

  /* Defrag page in "old" range. */
  bool is_empty = (src == end);
  if (!is_empty) {
    /* `page_cached_end` refers to the next empty slot.
     * Decrement by one to refer to the first slot we can defrag. */
    for (uint dst = end - 1; dst > src; dst--) {
      /* Find hole. */
      if (pages_cached_buf[dst].x != uint(-1)) {
        continue;
      }
      /* Update corresponding reference in tile. */
      shadow_page_cache_update_page_ref(src, dst);
      /* Move page. */
      pages_cached_buf[dst % max_page] = pages_cached_buf[src % max_page];
      pages_cached_buf[src % max_page] = uvec2(-1);

      find_first_valid(src, dst);
    }
  }

  end = pages_infos_buf.page_cached_next;
  /* Free pages in the "new" range (these are compact). */
  for (; additional_pages > 0 && src < end; additional_pages--, src++) {
    page_cached_free(src % max_page);
  }

  pages_infos_buf.page_cached_start = src;
  pages_infos_buf.page_cached_end = pages_infos_buf.page_cached_next;
  pages_infos_buf.page_alloc_count = 0;

  /* Wrap the cursor to avoid unsigned overflow. We do not do modulo arithmetic because it would
   * produce a 0 length buffer if the buffer is full. */
  if (pages_infos_buf.page_cached_start > max_page) {
    pages_infos_buf.page_cached_next -= max_page;
    pages_infos_buf.page_cached_start -= max_page;
    pages_infos_buf.page_cached_end -= max_page;
  }
}