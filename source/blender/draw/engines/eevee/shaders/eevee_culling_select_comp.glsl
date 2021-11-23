
/**
 * Select the visible items inside the active view and put them inside the sorting buffer.
 */

#pragma BLENDER_REQUIRE(common_debug_lib.glsl)
#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_geom_lib.glsl)
#pragma BLENDER_REQUIRE(common_intersection_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_light_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)

layout(local_size_x = CULLING_ITEM_BATCH) in;

layout(std430, binding = 0) readonly restrict buffer lights_buf
{
  LightData lights[];
};

layout(std430, binding = 1) restrict buffer culling_buf
{
  CullingData culling;
};

layout(std430, binding = 2) restrict buffer key_buf
{
  uint keys[];
};

void main()
{
  uint l_idx = gl_GlobalInvocationID.x;
  if (l_idx >= culling.items_count) {
    return;
  }

  LightData light = lights[l_idx];

  Sphere sphere;
  switch (light.type) {
    case LIGHT_SUN:
      sphere = Sphere(cameraPos, ViewFar * 2.0);
      break;
    case LIGHT_SPOT:
      /* TODO cone culling. */
    case LIGHT_RECT:
    case LIGHT_ELLIPSE:
    case LIGHT_POINT:
      sphere = Sphere(light._position, light.influence_radius_max);
      break;
  }

  if (intersect_view(sphere)) {
    uint index = atomicAdd(culling.visible_count, 1);
    keys[index] = l_idx;
  }
}
