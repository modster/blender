
/**
 * Virtual shadowmapping: Visibility phase for tilemaps.
 * During this phase we compute the visibility of each tile for the active view frustum.
 * TODO(fclem) Could also test visibility against Z buffer (would help in interiors space).
 */

#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_intersection_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_geom_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shadow_tilemap_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)

layout(local_size_x = SHADOW_TILEMAP_RES, local_size_y = SHADOW_TILEMAP_RES) in;

layout(std430, binding = 0) readonly buffer tilemaps_buf
{
  ShadowTileMapData tilemaps[];
};

layout(r32ui) restrict uniform uimage2D tilemaps_img;

void main()
{
  ShadowTileMapData tilemap = tilemaps[gl_GlobalInvocationID.z];
  ivec2 tile_co = ivec2(gl_GlobalInvocationID.xy);

  bool is_intersecting;
  int lod_visible_min = 0;
  int lod_visible_max = 0;

  if (tilemap.is_cubeface) {
    Pyramid shape = shadow_tilemap_cubeface_bounds(tilemap, tile_co, ivec2(1));

    is_intersecting = intersect_view(shape);

    if (is_intersecting && (tilemap.cone_angle_cos > -1.0)) {
      /* Reject tile not in spot light cone angle. */
      vec3 tile_dir = normalize((shape.corners[3] - shape.corners[1]) * 0.5 +
                                (shape.corners[1] - shape.corners[0]));
      /* cone_angle_cos is already biased to include the maximum tile cone half angle. */
      if (dot(tilemap.cone_direction, tile_dir) < tilemap.cone_angle_cos) {
        is_intersecting = false;
      }
    }

    if (is_intersecting) {
      /* Test minimum receiver distance and compute min and max visible LOD.  */
      float len;
      vec3 tile_center = shape.corners[1] + shape.corners[3] * 0.5;
      vec3 corner_vec = normalize_len(tile_center - shape.corners[0], len);
      float projection_len = dot(corner_vec, cameraPos - shape.corners[0]);
      vec3 nearest_receiver = corner_vec * min(len, projection_len);

      lod_visible_min = shadow_punctual_lod_level(distance(nearest_receiver, cameraPos));
      /* FIXME(fclem): This should be computed using the farthest intersection with the view.  */
      lod_visible_max = SHADOW_TILEMAP_LOD;

      lod_visible_max = clamp(lod_visible_max, 0, SHADOW_TILEMAP_LOD);
      lod_visible_min = clamp(lod_visible_min, 0, SHADOW_TILEMAP_LOD);
    }

    /* Bitmap of intersection tests. Use one uint per row. */
    shared uint intersect_map[SHADOW_TILEMAP_RES];

    /* Number of lod0 tiles covered by the current lod level (in one dimension). */
    uint lod_stride = 1u;
    uint lod_size = uint(SHADOW_TILEMAP_RES);
    for (int lod = 1; lod <= SHADOW_TILEMAP_LOD; lod++) {
      lod_size >>= 1;
      lod_stride <<= 1;

      barrier();
      if (is_intersecting && lod >= lod_visible_min && lod <= lod_visible_max) {
        atomicOr(intersect_map[tile_co.y], (1u << tile_co.x));
      }
      else {
        atomicAnd(intersect_map[tile_co.y], ~(1u << tile_co.x));
      }
      /* Control flow is uniform inside a workgroup. */
      barrier();

      if (all(lessThan(tile_co, ivec2(lod_size)))) {
        uint col_mask = bit_field_mask(lod_stride, lod_stride * tile_co.x);
        bool visible = false;
        uint row = lod_stride * tile_co.y;
        uint row_max = lod_stride + row;
        for (; row < row_max && !visible; row++) {
          visible = (intersect_map[row] & col_mask) != 0;
        }
        if (visible) {
          shadow_tile_set_flag(tilemaps_img, tile_co, lod, tilemap.index, SHADOW_TILE_IS_VISIBLE);
        }
      }
    }
  }
  else {
    /* TODO(fclem): We can save a few tile more by shaping the BBoxes in depth based on their
     * distance to the center. */
    Box shape = shadow_tilemap_clipmap_bounds(tilemap, tile_co, ivec2(1));

    is_intersecting = intersect_view(shape);

    if (is_intersecting) {
      /* Reject tiles not in view distance. */
      float tile_dist = length(vec2(tile_co - (SHADOW_TILEMAP_RES / 2)) + 0.5);
      if (tile_dist > (SHADOW_TILEMAP_RES / 2) + M_SQRT2 * 0.5) {
        is_intersecting = false;
      }
    }
  }

  if (is_intersecting && lod_visible_min == 0) {
    shadow_tile_set_flag(tilemaps_img, tile_co, 0, tilemap.index, SHADOW_TILE_IS_VISIBLE);
  }
}