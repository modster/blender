
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)

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
  return page.y * uint(SHADOW_PAGE_PER_ROW) + page.x;
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

/** \a unormalized_uv is the uv coordinates for the whole tilemap [0..SHADOW_TILEMAP_RES]. */
vec2 shadow_page_uv_transform(uvec2 page, uint lod, vec2 unormalized_uv)
{
  vec2 page_texel = fract(unormalized_uv / float(1u << lod));
  /* Assumes atlas is squared. */
  return (vec2(page) + page_texel) / vec2(SHADOW_PAGE_PER_ROW);
}
