
#pragma BLENDER_REQUIRE(common_math_lib.glsl)

#define SHADOW_PAGE_IS_CACHED (1u << 17u)
#define SHADOW_PAGE_IS_FREE (1u << 18u)
#define SHADOW_PAGE_IS_USED (1u << 19u)
#define SHADOW_PAGE_IN_CACHE_HEAP (1u << 20u)

/** \a unormalized_uv is the uv coordinates for the whole tilemap [0..SHADOW_TILEMAP_RES]. */
vec2 shadow_page_uv_transform(uvec2 page, uint lod, vec2 unormalized_uv)
{
  vec2 page_texel = fract(unormalized_uv / float(1u << lod));
  /* Fix float imprecision that can make some pixel sample the wrong page. */
  page_texel *= 0.999999;
  /* Assumes atlas is squared. */
  return (vec2(page) + page_texel) / vec2(SHADOW_PAGE_PER_ROW);
}
