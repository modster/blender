
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)

IN_OUT AABBInterface
{
  flat vec2 center;
  flat vec2 half_extent;
  flat int tilemap_index;
}
interp;

/* Conservatively rasterize the AABB. */
bool aabb_raster(vec2 fragcoord)
{
  /* TODO(fclem): Potentially more saving can be done here. */
  fragcoord -= (interp.center * 0.5 + 0.5) * float(SHADOW_TILEMAP_RES);
  return all(lessThan(abs(fragcoord), interp.half_extent * float(SHADOW_TILEMAP_RES)));
}
