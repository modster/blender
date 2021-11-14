
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)

/**
 * HACK: Use last member to store the heap usage to avoid alignment.
 * Note that this stores the index to the last element.
 */
#define SHADOW_PAGE_HEAP_LAST_USED(heap) heap[SHADOW_MAX_PAGE - 1]

/** Decoded page data structure. */
struct ShadowPageData {
  /** Tile inside the tilemap atlas. */
  ivec2 tile;
};

#define SHADOW_PAGE_NO_DATA 0xFFFFFFFF

uvec2 shadow_page_from_index(ShadowPagePacked index)
{
  return uvec2(index % SHADOW_PAGE_PER_ROW, index / SHADOW_PAGE_PER_ROW);
}

uint shadow_page_to_index(uvec2 page)
{
  return page.y * SHADOW_PAGE_PER_ROW + page.x;
}

ShadowPageData shadow_page_data_unpack(ShadowPagePacked data)
{
  ShadowPageData page;
  page.tile.x = data & 0xFFF;
  page.tile.y = (data >> 12) & 0xFFF;
  return page;
}

ShadowPagePacked shadow_page_data_pack(ShadowPageData page)
{
  ShadowPagePacked data;
  data = page.tile.x;
  data |= page.tile.y << 12;
  return data;
}
