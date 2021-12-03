
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

#pragma BLENDER_REQUIRE(common_intersection_lib.glsl)
#pragma BLENDER_REQUIRE(common_view_lib.glsl)
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

vec3 safe_project(ShadowTileMapData tilemap, inout int clipped, vec3 v)
{
  vec4 tmp = tilemap.tilemat * vec4(v, 1.0);
  /* Detect case when point is behind the camera. */
  clipped += int(tmp.w < 0.0);
  return tmp.xyz / tmp.w;
}

void main()
{
  ShadowTileMapData tilemap = tilemaps[gl_GlobalInvocationID.z];

  /* Bitmap of tile intersection tests. Use one uint per row for each LOD. */
  shared uint flag_map[SHADOW_TILEMAP_RES * 2];
  if (gl_LocalInvocationID.x == 0u) {
    for (int i = 0; i < SHADOW_TILEMAP_RES * 2; i++) {
      flag_map[i] = 0u;
    }
  }
  barrier();

  Pyramid frustum;
  if (tilemap.is_cubeface) {
    frustum = shadow_tilemap_cubeface_bounds(tilemap, ivec2(0), ivec2(SHADOW_TILEMAP_RES));
  }

  int iter = divide_ceil_i(aabb_len, int(gl_WorkGroupSize.x));
  for (int i = 0; i < iter; i++) {
    int aabb_index = i * int(gl_WorkGroupSize.x) + int(gl_GlobalInvocationID.x);
    if (aabb_index >= aabb_len) {
      break;
    }
    AABB aabb = aabbs[aabb_index];
    /* Avoid completely flat object disapearing. */
    aabb.max += 1e-6;
    aabb.min -= 1e-6;

    Box box = to_box(aabb);

#ifdef TAG_USAGE
    /* Skip non visible objects. */
    if (!intersect_view(box)) {
      continue;
    }
#endif

#ifdef TAG_UPDATE
    // drw_debug(aabb, vec4(0, 1, 0, 1));
#else /* TAG_USAGE */
    // drw_debug(aabb, vec4(1, 1, 0, 1));
#endif

    int lod_min = tilemap.is_cubeface ? SHADOW_TILEMAP_LOD : 0;
    int lod_max = 0;
    int clipped = 0;
    /* TODO(fclem) project bbox instead of AABB. */
    /* NDC space post projection [-1..1] (unclamped). */
    AABB aabb_ndc = init_min_max();
    for (int v = 0; v < 8; v++) {
      merge(aabb_ndc, safe_project(tilemap, clipped, box.corners[v]));

#ifdef TAG_USAGE
      if (tilemap.is_cubeface) {
        /* FIXME(fclem) this will fail if camera is inside the box.  */
        int lod_visible = shadow_punctual_lod_level(distance(cameraPos, box.corners[v]));
        lod_min = min(lod_min, lod_visible);
        lod_max = max(lod_max, lod_visible);
      }
#endif
    }
    lod_max = clamp(lod_max, 0, SHADOW_TILEMAP_LOD);
    lod_min = clamp(lod_min, 0, SHADOW_TILEMAP_LOD);

#ifdef TAG_UPDATE
    /* Update tag all LODs. */
    lod_max = SHADOW_TILEMAP_LOD;
    lod_min = 0;
#endif

    if (tilemap.is_cubeface) {
      if (clipped == 8) {
        /* All verts are behind the camera. */
        continue;
      }
      else if (clipped > 0) {
        /* Not all verts are behind the near clip plane. */
        if (intersect(frustum, box)) {
          /* We cannot correctly handle this case so we fallback by covering the whole view. */
          aabb_ndc.max = vec3(vec2(SHADOW_TILEMAP_RES), 1.0);
          aabb_ndc.min = vec3(0.0, 0.0, -1.0);
        }
        else {
          /* Still out of the frustum. Ignore. */
          continue;
        }
      }
      else {
        /* Reject false positive when box is on the side of the frustum but fail other tests. */
        /* AABB of the entire light is enough for this case. */
        AABB aabb_light = AABB(frustum.corners[0] - vec3(tilemap._punctual_distance),
                               frustum.corners[0] + vec3(tilemap._punctual_distance));
        if (!intersect(aabb, aabb_light)) {
          continue;
        }
      }
    }

    /* Discard if the bbox does not touch the rendering frustum. */
#ifdef TAG_UPDATE
    const float min_depth = -1.0;
    const float max_depth = 1.0;
#else /* TAG_USAGE */
    float max_depth = tilemap._max_usage_depth;
    float min_depth = tilemap._min_usage_depth;
#endif
    AABB aabb_tag;
    const AABB aabb_map = AABB(vec3(0.0, 0.0, min_depth),
                               vec3(vec2(SHADOW_TILEMAP_RES) - 1e-6, max_depth));
    if (!intersect(aabb_map, aabb_ndc, aabb_tag)) {
      continue;
    }

    /* Raster the Box. */
    ivec2 range_min = ivec2(aabb_tag.min.xy) >> lod_min;
    ivec2 range_max = ivec2(aabb_tag.max.xy) >> lod_min;

    for (int lod = lod_min; lod <= lod_max; lod++) {
      for (int y = range_min.y; y <= range_max.y; y++) {
        int flag_idx = (SHADOW_TILEMAP_RES >> lod) + y;
        uint row_bits = bit_field_mask(range_max.x - range_min.x + 1, range_min.x);
        atomicOr(flag_map[flag_idx], row_bits);
      }
      range_min >>= 1;
      range_max >>= 1;
    }
  }

  barrier();

  if (gl_LocalInvocationID.x == 0u) {
#ifdef TAG_UPDATE
    const uint flag = SHADOW_TILE_DO_UPDATE;
#else /* TAG_USAGE */
    const uint flag = SHADOW_TILE_IS_USED;
#endif
    int lod_max = tilemap.is_cubeface ? SHADOW_TILEMAP_LOD : 0;
    /* Number of lod0 tiles covered by the current lod level (in one dimension). */
    uint lod_size = uint(SHADOW_TILEMAP_RES);
    /* TODO(fclem): Could use multiple thread to set flag. */
    for (int lod = 0; lod <= lod_max; lod++, lod_size >>= 1) {
      for (int y = 0; y < lod_size; y++) {
        int flag_idx = (SHADOW_TILEMAP_RES >> lod) + y;
        uint row_bits = flag_map[flag_idx];
        while (row_bits != 0u) {
          int x = findLSB(row_bits);
          row_bits &= ~1u << uint(x);
          shadow_tile_set_flag(tilemaps_img, ivec2(x, y), lod, tilemap.index, flag);
        }
      }
    }
  }
}