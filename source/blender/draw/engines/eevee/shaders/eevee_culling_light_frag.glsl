
/**
 * 2D Culling pass for lights.
 * We iterate over all items and check if they intersect with the tile frustum.
 */

#pragma BLENDER_REQUIRE(eevee_culling_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_culling_iter_lib.glsl)

layout(std140) uniform lights_block
{
  LightData lights[CULLING_ITEM_BATCH];
};

layout(std140) uniform lights_culling_block
{
  CullingData culling;
};

in vec4 uvcoordsvar;

layout(location = 0) out uvec4 out_items_bits;

void main(void)
{
  CullingTile tile = culling_tile_get(culling);

  out_items_bits = uvec4(0);
  ITEM_FOREACH_BEGIN_NO_CULL (culling, l_idx) {
    LightData light = lights[l_idx];

    bool intersect_tile = true;
    switch (light.type) {
      case LIGHT_SPOT:
        /* TODO cone culling. */
      case LIGHT_RECT:
      case LIGHT_ELLIPSE:
      case LIGHT_POINT:
        Sphere sphere = Sphere(light._position, light.influence_radius_max);
        intersect_tile = culling_sphere_tile_isect(sphere, tile);
        break;
      default:
        break;
    }

    if (intersect_tile) {
      out_items_bits[l_idx / 32u] |= 1u << (l_idx % 32u);
    }
  }
  ITEM_FOREACH_END
}