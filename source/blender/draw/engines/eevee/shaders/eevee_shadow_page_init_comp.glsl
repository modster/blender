
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

layout(std430, binding = 3) restrict writeonly buffer pages_infos_buf
{
  ShadowPagesInfoData infos;
};

void main()
{
  infos.page_free_next = SHADOW_MAX_PAGE - 1;
  infos.page_free_next_prev = 0;
  infos.page_updated_count = 0;

  pages_free[gl_GlobalInvocationID.x] = ShadowPagePacked(gl_GlobalInvocationID.x);

#ifdef SHADOW_DEBUG_PAGE_ALLOCATION_ENABLED
  pages[gl_GlobalInvocationID.x] = SHADOW_PAGE_NO_DATA;
#endif
}