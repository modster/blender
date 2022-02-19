
/**
 * Virtual shadowmapping: Init page buffer.
 *
 * All pages are always owned by tiles. This step init all owners.
 * This avoid mapping the buffer to host memory.
 */

#pragma BLENDER_REQUIRE(eevee_shadow_page_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shadow_tilemap_lib.glsl)

void main()
{
  if (gl_GlobalInvocationID == uvec3(0)) {
    pages_infos_buf.page_free_count = SHADOW_MAX_PAGE;
    pages_infos_buf.page_alloc_count = 0;
    pages_infos_buf.page_cached_next = 0u;
    pages_infos_buf.page_cached_start = 0u;
    pages_infos_buf.page_cached_end = 0u;
  }

  uint page_index = gl_GlobalInvocationID.x;

  ivec2 texel = ivec2(page_index % SHADOW_PAGE_PER_ROW, page_index / SHADOW_PAGE_PER_ROW);
  pages_free_buf[page_index] = packUvec2x16(uvec2(texel));
  pages_cached_buf[page_index * 2 + 0] = uvec2(-1);
  pages_cached_buf[page_index * 2 + 1] = uvec2(-1);
}