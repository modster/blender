
/**
 * Virtual shadowmapping: AABB rendering
 * Making entity checks on CPU can be an intensive task if scene is really complex.
 * Some entities may need to tag certain shadow pages to be updated or needed.
 */

/**
 * TODO(fclem) : Future plans. Do not rely on rasterization and use a single compute shader for
 * both visibility and usage. Une one thread group per tilemap and each thread compute one AABB.
 * Use groupshared memory to hold a bitmap of the result. Each thread "rasterize" onto the bitmap
 * using atomicOr. Iterate through all AABBs using this threagroup and can early out if all tiles
 * are tagged. Could even compute AABB of each batch of 64 to skip an entire batch.
 *
 * This would avoid relying on arbitrary point size support and be more performant.
 */

#pragma BLENDER_REQUIRE(common_math_geom_lib.glsl)
#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_debug_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)
#pragma BLENDER_REQUIRE(eevee_shadow_tilemap_lib.glsl)

layout(local_size_x = SHADOW_AABB_TAG_GROUP_SIZE) in;

layout(std430, binding = 0) readonly buffer tilemaps_buf
{
  ShadowTileMapData tilemaps[];
};

layout(std430, binding = 1) readonly buffer aabb_buf
{
  AABB aabbs[];
};

layout(r32ui) restrict uniform uimage2D tilemaps_img;

uniform int aabb_len;

vec3 safe_project(inout int clipped, vec3 v)
{
  vec4 tmp = tilemaps[gl_GlobalInvocationID.z].persmat * vec4(v, 1.0);
  /* Detect case when point is behind the camera. */
  clipped += int(tmp.w < 0.0);
  return tmp.xyz / tmp.w;
}

void main()
{
  float tile_ndc_size = shadow_tile_coord_to_ndc(ivec2(1)).x -
                        shadow_tile_coord_to_ndc(ivec2(0)).x;

  int iter = (aabb_len + (int(gl_WorkGroupSize.x) - 1)) / int(gl_WorkGroupSize.x);
  for (int i = 0; i < iter; i++) {
    int aabb_index = i * int(gl_WorkGroupSize.x) + int(gl_GlobalInvocationID.x);
    if (aabb_index >= aabb_len) {
      break;
    }

    AABB aabb = aabbs[aabb_index];

#ifdef TAG_UPDATE
    drw_debug(aabb, vec4(0, 1, 0, 1));
#else /* TAG_USAGE */
    // drw_debug(aabb, vec4(1, 1, 0, 1));
#endif

    Box box = to_box(aabb);
    int clipped = 0;
    /* TODO(fclem) project bbox instead of AABB. */
    /* NDC space post projection [-1..1] (unclamped). */
    AABB aabb_ndc = init_min_max();
    for (int v = 0; v < 8; v++) {
      merge(aabb_ndc, safe_project(clipped, box.corners[v]));
    }

    if (clipped == 8) {
      /* All verts are behind the camera. */
      continue;
    }

    if (clipped > 0) {
      /* Not all verts are behind the near clip plane. */
      /* We cannot correctly handle this case so we fallback to making them cover the whole view.
       */
      /* TODO/FIXME(fclem) This clearly needs a better solution as it is a common case.
       * Idea, pyramid vs. box test. */
      aabb_ndc.max = vec3(1.0);
      aabb_ndc.min = vec3(-1.0);
    }

    /* Discard if the bbox does not touch the rendering frustum in the depth axis. */
    if (aabb_ndc.max.z < -1.0 || aabb_ndc.min.z > 1.0) {
      continue;
    }

    vec3 center = (aabb_ndc.min + aabb_ndc.max) * 0.5;

    /* Raster the Box. */
    /* Could only visit visible tiles. */

    AABB aabb_map = AABB(vec3(-1), vec3(1));
    if (!intersect(aabb_map, aabb_ndc, aabb_ndc)) {
      continue;
    }

    ivec2 range_min, range_max;
    range_min = ivec2((aabb_ndc.min.xy + 1.0) * float(SHADOW_TILEMAP_RES / 2));
    range_max = ivec2((aabb_ndc.max.xy + 1.0) * float(SHADOW_TILEMAP_RES / 2));
    range_min = min(range_min, ivec2(SHADOW_TILEMAP_RES - 1));
    range_max = min(range_max, ivec2(SHADOW_TILEMAP_RES - 1));

    /* OPTI(fclem): Could only test tiles not already tagged. */
    for (int y = range_min.y; y <= range_max.y; y++) {
      for (int x = range_min.x; x <= range_max.x; x++) {
        ivec2 tile_co = ivec2(x, y);
        AABB aabb_tile = AABB(vec3(tile_ndc_size * vec2(tile_co + 0) - 1.0, -1.0),
                              vec3(tile_ndc_size * vec2(tile_co + 1) - 1.0, 1.0));
#ifdef TAG_UPDATE
        const uint flag = SHADOW_TILE_DO_UPDATE;
#else /* TAG_USAGE */
        const uint flag = SHADOW_TILE_IS_USED;
#endif
        int tilemap_index = tilemaps[gl_GlobalInvocationID.z].index;
        /* TODO(fclem) Reduce atomic contention. */
        shadow_tile_set_flag(tilemaps_img, tile_co, tilemap_index, flag);
      }
    }
  }
}