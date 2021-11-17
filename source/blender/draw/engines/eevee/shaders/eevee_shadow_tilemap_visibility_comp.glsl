
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

  if (tilemap.is_cubeface) {
    Pyramid shape = shadow_tilemap_cubeface_bounds(tilemap, tile_co, ivec2(1));

    is_intersecting = intersect_view(shape);

    if (is_intersecting && (tilemap.cone_angle_cos > -1.0)) {
      /* Reject tile not in spot light cone angle. */
      vec3 tile_dir = normalize((shape.corners[3] - shape.corners[1]) * 0.5 +
                                (shape.corners[1] - shape.corners[0]));
      /* cone_angle_cos is already bias to include the maximum tile cone half angle. */
      if (dot(tilemap.cone_direction, tile_dir) < tilemap.cone_angle_cos) {
        is_intersecting = false;
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

  if (is_intersecting) {
    shadow_tile_set_flag(tilemaps_img, tile_co, tilemap.index, SHADOW_TILE_IS_VISIBLE);
  }
  /* TODO Do Mips for cubemaps. Could do recursive downsampling using groupshared memory. */
}