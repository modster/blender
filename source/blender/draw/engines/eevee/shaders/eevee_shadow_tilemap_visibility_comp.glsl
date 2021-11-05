
/**
 * Virtual shadowmapping: Visibility phase for tilemaps.
 * During this phase we compute the visibility of each tile for the active view frustum.
 */

#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shadow_tilemap_lib.glsl)

layout(local_size_x = SHADOW_TILEMAP_RES, local_size_y = SHADOW_TILEMAP_RES) in;

layout(std430, binding = 0) readonly buffer tilemaps_block
{
  ShadowTileMapData tilemaps[];
};

layout(r32ui) restrict uniform uimage2D tilemaps_img;

bool tile_intersect_frustum()
{
  /* TODO(fclem) Finish */
  return true;
}

void main()
{
  ivec2 tile_co = ivec2(gl_GlobalInvocationID.xy);
  int tilemap_idx = int(gl_GlobalInvocationID.z);

  if (tile_intersect_frustum()) {
    shadow_tile_set_flag(tilemaps_img, tile_co, tilemap_idx, SHADOW_TILE_IS_VISIBLE);
  }
  /* TODO Do Mips for cubemaps. Could do recursive downsampling using groupshared memory. */
}