
/**
 * Virtual shadowmapping: Init page buffer.
 *
 * This avoid mapping the buffer to host memory.
 */

#pragma BLENDER_REQUIRE(eevee_shadow_page_lib.glsl)

layout(local_size_x = SHADOW_PAGE_PER_ROW) in;

layout(std430, binding = 1) restrict writeonly buffer pages_free_buf
{
  int pages_free[];
};

layout(std430, binding = 2) restrict writeonly buffer pages_buf
{
  ShadowPagePacked pages[];
};

void main()
{
  /* Note this also kindly sets SHADOW_PAGE_HEAP_LAST to the right amount (SHADOW_MAX_PAGE - 2). */
  uint free_page_index = min(gl_GlobalInvocationID.x, SHADOW_MAX_PAGE - 2);

  pages_free[gl_GlobalInvocationID.x] = int(free_page_index);
#ifdef SHADOW_DEBUG_PAGE_ALLOCATION_ENABLED
  pages[gl_GlobalInvocationID.x] = SHADOW_PAGE_NO_DATA;
#endif
}