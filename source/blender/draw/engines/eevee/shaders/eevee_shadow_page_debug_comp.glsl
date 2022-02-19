
/**
 * Virtual shadowmapping: Debug pages.
 *
 * Since pages are only existing if they are attached to a tilemap or the free list,
 * this shader will scan every possible position and create a debug map out of it.
 * This is nice to inspect the state of the page allocation during the pipeline.
 */

#pragma BLENDER_REQUIRE(eevee_shadow_page_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shadow_tilemap_lib.glsl)

/* Use this as custom channel viewer in renderdoc to inspect debug_img. */
#if 0

layout(binding = 2) uniform usampler2D texUInt2D;
in vec2 uv;
out vec4 color_out;

#  ifndef SHADOW_PAGE_IS_CACHED
#    error "Do not forget to copy the falgs too"
#  endif

void main()
{
  uint page = texture(texUInt2D, uv).x;

  uint user_count = page & 0xFFFFu;
  bool is_cached = (page & SHADOW_PAGE_IS_CACHED) != 0u;
  bool is_free = (page & SHADOW_PAGE_IS_FREE) != 0u;
  bool is_used = (page & SHADOW_PAGE_IS_USED) != 0u;
  bool in_cache_heap = (page & SHADOW_PAGE_IN_CACHE_HEAP) != 0u;

  const uint all_states = (SHADOW_PAGE_IS_CACHED | SHADOW_PAGE_IS_FREE | SHADOW_PAGE_IS_USED);

  vec3 col;
  if (bitCount(page & all_states) != 1) {
    col = vec3(1, 0, 0);
  }
  else if (is_cached && !in_cache_heap) {
    col = vec3(1, 0, 0);
  }
  else if (user_count != 1u) {
    col = vec3(1, 0, 0);
  }
  else if (is_cached) {
    col = vec3(1, 1, 0);
  }
  else if (is_used) {
    col = vec3(0, 1, 0);
  }
  else if (is_free) {
    col = vec3(0, 0.2, 1);
  }
  else {
    /* Error: Unknown state. */
    col = vec3(0);
  }

  color_out = vec4(col, 1.0);
}

#endif

void main()
{
  for (int y = 0; y < imageSize(debug_img).y / int(gl_WorkGroupSize.y); y++) {
    for (int x = 0; x < imageSize(debug_img).x / int(gl_WorkGroupSize.x); x++) {
      ivec2 co = ivec2(x, y) * ivec2(gl_WorkGroupSize.xy) + ivec2(gl_LocalInvocationID.xy);
      imageStore(debug_img, co, uvec4(0));
    }
  }

  barrier();

  for (int y = 0; y < imageSize(tilemaps_img).y / int(gl_WorkGroupSize.y); y++) {
    for (int x = 0; x < imageSize(tilemaps_img).x / int(gl_WorkGroupSize.x); x++) {
      ivec2 co = ivec2(x, y) * ivec2(gl_WorkGroupSize.xy) + ivec2(gl_LocalInvocationID.xy);
      ShadowTileData tile = shadow_tile_data_unpack(imageLoad(tilemaps_img, co).x);

      if (tile.is_allocated) {
        imageAtomicOr(debug_img, ivec2(tile.page), SHADOW_PAGE_IS_USED);
        /* User count. */
        imageAtomicAdd(debug_img, ivec2(tile.page), 1u);
      }
      if (tile.is_cached) {
        uvec2 page = unpackUvec2x16(pages_cached_buf[tile.cache_index].x);
        imageAtomicOr(debug_img, ivec2(page), SHADOW_PAGE_IS_CACHED);
      }
    }
  }

  if (gl_GlobalInvocationID == uvec3(0)) {
    for (int x = 0; x < pages_infos_buf.page_free_count; x++) {
      if (pages_free_buf[x] != uint(-1)) {
        uvec2 page = unpackUvec2x16(pages_free_buf[x]);
        imageAtomicOr(debug_img, ivec2(page), SHADOW_PAGE_IS_FREE);
        /* User count. */
        imageAtomicAdd(debug_img, ivec2(page), 1u);
      }
    }

    for (int x = 0; x < SHADOW_MAX_PAGE; x++) {
      if (pages_cached_buf[x].x != uint(-1)) {
        uvec2 page = unpackUvec2x16(pages_cached_buf[x].x);
        imageAtomicOr(debug_img, ivec2(page), SHADOW_PAGE_IN_CACHE_HEAP);
        /* User count. */
        imageAtomicAdd(debug_img, ivec2(page), 1u);
      }
    }
  }
}