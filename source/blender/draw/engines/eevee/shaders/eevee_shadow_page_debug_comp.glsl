
/**
 * Virtual shadowmapping: Debug pages.
 *
 * Since pages are only existing if they are attached to a tilemap or the free list,
 * this shader will scan every possible position and create a debug map out of it.
 * This is nice to inspect the state of the page allocation during the pipeline.
 */

#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shadow_page_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shadow_tilemap_lib.glsl)

layout(local_size_x = 8, local_size_y = 8) in;

layout(std430, binding = 1) readonly restrict buffer pages_free_buf
{
  uint free_page_owners[];
};

layout(r32ui) restrict uniform uimage2D tilemaps_img;

layout(r32ui) restrict uniform uimage2D debug_img;

/* Use this as custom channel viewer in renderdoc to inspect debug_img. */
#if 0

layout(binding = 2) uniform usampler2D texUInt2D;
in vec2 uv;
out vec4 color_out;

void main()
{
  uint page = texture(texUInt2D, uv).x;

  bool error = (page & 0xFFFFu) != 1u;
  bool is_cached = (page & SHADOW_PAGE_IS_CACHED) != 0u;
  bool is_needed = (page & SHADOW_PAGE_IS_NEEDED) != 0u;
  bool in_heap = (page & SHADOW_PAGE_IN_FREE_HEAP) != 0u;
  error = error || (is_cached && !in_heap);
  error = error || (is_needed && is_cached);

  color_out = vec4(error, is_cached, is_needed, in_heap);
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

  /* TODO(fclem): We only scan the first line of tilemap, otherwise it is too slow.
   * Finish and do it properly one day... */
  for (int y = 0; y < imageSize(tilemaps_img).y / int(gl_WorkGroupSize.y); y++) {
    for (int x = 0; x < imageSize(tilemaps_img).x / int(gl_WorkGroupSize.x); x++) {
      ivec2 co = ivec2(x, y) * ivec2(gl_WorkGroupSize.xy) + ivec2(gl_LocalInvocationID.xy);
      ShadowTileData tile = shadow_tile_data_unpack(imageLoad(tilemaps_img, co).x);

      if (tile.is_allocated) {
        /* User count. */
        imageAtomicAdd(debug_img, ivec2(tile.page), 1u);
      }
    }
  }

  barrier();

  for (int y = 0; y < imageSize(tilemaps_img).y / int(gl_WorkGroupSize.y); y++) {
    for (int x = 0; x < imageSize(tilemaps_img).x / int(gl_WorkGroupSize.x); x++) {
      ivec2 co = ivec2(x, y) * ivec2(gl_WorkGroupSize.xy) + ivec2(gl_LocalInvocationID.xy);
      ShadowTileData tile = shadow_tile_data_unpack(imageLoad(tilemaps_img, co).x);

      if (tile.is_allocated) {
        imageAtomicOr(debug_img, ivec2(tile.page), SHADOW_PAGE_ALLOCATED);
        if (tile.is_cached) {
          imageAtomicOr(debug_img, ivec2(tile.page), SHADOW_PAGE_IS_CACHED);
          /* Verify reference. */
          ivec2 ref = ivec2(unpackUvec2x16(free_page_owners[tile.free_page_owner_index]));
          if (ref == co) {
            imageAtomicOr(debug_img, ivec2(tile.page), SHADOW_PAGE_IN_FREE_HEAP);
          }
        }
        if (tile.is_used && tile.is_visible) {
          imageAtomicOr(debug_img, ivec2(tile.page), SHADOW_PAGE_IS_NEEDED);
        }
        if (tile.do_update) {
          imageAtomicOr(debug_img, ivec2(tile.page), SHADOW_PAGE_DO_UPDATE);
        }
      }
    }
  }

#if 0
  for (int x = 0; x < SHADOW_MAX_PAGE; x++) {
    if (free_page_owners[x] != uint(-1)) {
      uvec2 owner = unpackUvec2x16(free_page_owners[x]);
      uvec2 page = shadow_tile_data_unpack(imageLoad(tilemaps_img, ivec2(owner)).x).page;
      /* User count. */
      imageAtomicAdd(debug_img, ivec2(page), 1u);

      imageAtomicOr(debug_img, ivec2(page), SHADOW_PAGE_IN_FREE_HEAP);
    }
  }
#endif
}