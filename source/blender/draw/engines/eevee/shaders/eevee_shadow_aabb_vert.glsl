
/**
 * Virtual shadowmapping: AABB rendering
 * Making entity checks on CPU can be an intensive task if scene is really complexe.
 * Some entities may need to tag certain shadow pages to be updated or needed.
 * To tag efficiently, we render a list of points using the object Axis Aligned Bounding Box
 * to get a first approximation of all the tiles the entity is touching.
 * The fragment shader then refine the check for every tiles it is invoked for.
 * Important detail is that we do a conservative rasterization of the Bound box to not miss
 * any tile it could be touching.
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

#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shadow_aabb_interface_lib.glsl)

layout(std430, binding = 0) readonly buffer tilemaps_block
{
  ShadowTileMapData tilemaps[];
};

in vec3 aabb_min;
in vec3 aabb_max;

struct AABB {
  vec3 min, max;
};

vec3 safe_project(inout int clipped, vec3 v)
{
  vec4 tmp = tilemaps[interp.tilemap_index].persmat * vec4(v, 1.0);
  /* Detect case when point is behind the camera. */
  clipped += int(tmp.w < 0.0);
  return tmp.xyz / tmp.w;
}

void merge(inout AABB aabb, vec3 v)
{
  /* Fix for perspective where  */
  aabb.min = min(aabb.min, v);
  aabb.max = max(aabb.max, v);
}

void main()
{
  /* Default: We set the vertex at the camera origin to generate 0 fragments.
   * Avoid undefined behavior. */
  gl_Position = vec4(0.0, 0.0, -3e36, 0.0);
  gl_PointSize = 1.0;

  interp.tilemap_index = gl_InstanceID;

  /* NDC space post projection [-1..1] (unclamped). */
  AABB aabb;
  aabb.min = vec3(1.0e30);
  aabb.max = vec3(-1.0e30);

  int clipped = 0;

  /* Compute bounds of the projected aabb. */
  merge(aabb, safe_project(clipped, aabb_min));
  merge(aabb, safe_project(clipped, vec3(aabb_max.x, aabb_min.y, aabb_min.z)));
  merge(aabb, safe_project(clipped, vec3(aabb_min.x, aabb_max.y, aabb_min.z)));
  merge(aabb, safe_project(clipped, vec3(aabb_min.x, aabb_min.y, aabb_max.z)));
  merge(aabb, safe_project(clipped, vec3(aabb_max.x, aabb_max.y, aabb_min.z)));
  merge(aabb, safe_project(clipped, vec3(aabb_min.x, aabb_max.y, aabb_max.z)));
  merge(aabb, safe_project(clipped, vec3(aabb_max.x, aabb_min.y, aabb_max.z)));
  merge(aabb, safe_project(clipped, aabb_max));

  vec3 center = (aabb.min + aabb.max) * 0.5;

  if (clipped == 8) {
    /* All verts are behind the camera. */
    return;
  }

  if (clipped > 0) {
    /* Not all verts are behind the near clip plane. */
    /* We cannot correctly handle this case so we fallback to making them cover the whole view. */
    /* TODO/FIXME(fclem) This clearly needs a better solution as it is a common case. */
    aabb.max = vec3(1.0);
    aabb.min = vec3(-1.0);
  }

  /* Discard if the bbox does not touch the rendering frustum in the depth axis. */
  if (aabb.max.z < -1.0 || aabb.min.z > 1.0) {
    return;
  }

  interp.center = (aabb.max + aabb.min).xy * 0.5;
  interp.half_extent = (aabb.max - aabb.min).xy * 0.5;

  gl_Position = vec4(interp.center, 0.0, 1.0);
  gl_PointSize = max_v2(interp.half_extent) * float(SHADOW_TILEMAP_RES) + 1.01;
}