
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

uniform float tilemap_pixel_radius;
uniform float screen_pixel_radius_inv;

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
      vec3 tile_center = (shape.corners[1] + shape.corners[3]) * 0.5;
      vec3 tile_center_dir = normalize_len(tile_center - shape.corners[0], len);
      /* Project the tile center to the frustum and compare the shadow texel density at this
       * position since this is where the density ratio will be the lowest (meanning the highest
       * LOD). NOTE: There is some inacuracy because we only project one point instead of
       * projecting each individual pixels.  */
      for (int p = 0; p < 6; p++) {
        float facing = dot(tile_center_dir, -frustum_planes[p].xyz);
        float d = line_plane_intersect_dist(shape.corners[0], tile_center_dir, frustum_planes[p]);
        if (d > 0.0 && facing > 0.0) {
          len = min(d, len);
        }
      }
      vec3 nearest_receiver = shape.corners[0] + tile_center_dir * len;
      /* How much a shadow map pixel covers a final image pixel. */
      float footprint_ratio = len * (tilemap_pixel_radius * screen_pixel_radius_inv);
      /* Project the radius to the screen. 1 unit away from the camera the same way
       * pixel_world_radius_inv was computed. Not needed in orthographic mode. */
      bool is_persp = (ProjectionMatrix[3][3] == 0.0);
      if (is_persp) {
        footprint_ratio /= distance(nearest_receiver, cameraPos);
      }

#if 0 /* DEBUG */
      if (gl_GlobalInvocationID.z == 0u) {
        vec4 green = vec4(0, 1, 0, 1);
        vec4 yellow = vec4(1, 1, 0, 1);
        vec4 red = vec4(1, 0, 0, 1);
        float dist_fac = (is_persp) ? distance(nearest_receiver, cameraPos) : 1.0;
        drw_debug_point(nearest_receiver, 128.0 * dist_fac / screen_pixel_radius_inv, green);
        drw_debug_point(shape.corners[0] + tile_center_dir, tilemap_pixel_radius * 128.0, red);
        drw_debug_point(nearest_receiver, len * tilemap_pixel_radius * 128.0, yellow);
      }
#endif

      lod_visible_min = int(ceil(-log2(footprint_ratio)));
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