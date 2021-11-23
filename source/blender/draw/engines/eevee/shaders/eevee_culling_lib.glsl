
#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)

/* ---------------------------------------------------------------------- */
/** \name Intersection Tests
 * \{ */

struct Cone {
  vec3 direction;
  float angle_cos;
};

struct Cylinder {
  /* Assume Z axis as direction. */
  vec3 center;
  float radius;
};

struct Frustum {
  vec4 planes[4];
};

struct CullingTile {
  Frustum frustum;
  Cone cone;
};

bool culling_sphere_cone_isect(Sphere sphere, Cone cone)
{
  /**
   * Following "Improve Tile-based Light Culling with Spherical-sliced Cone"
   * by Eric Zhang
   * https://lxjk.github.io/2018/03/25/Improve-Tile-based-Light-Culling-with-Spherical-sliced-Cone.html
   */
  float sphere_distance = length(sphere.center);
  float sphere_sin = saturate(sphere.radius / sphere_distance);
  float sphere_cos = sqrt(1.0 - sphere_sin * sphere_sin);
  float cone_aperture_sin = sqrt(1.0 - cone.angle_cos * cone.angle_cos);

  float cone_sphere_center_cos = dot(sphere.center / sphere_distance, cone.direction);
  /* cos(A+B) = cos(A) * cos(B) - sin(A) * sin(B). */
  float cone_sphere_angle_sum_cos = (sphere.radius > sphere_distance) ?
                                        -1.0 :
                                        (cone.angle_cos * sphere_cos -
                                         cone_aperture_sin * sphere_sin);

  /* Comparing cosines instead of angles since we are interested
   * only in the monotonic region [0 .. M_PI / 2]. This saves costly acos() calls. */
  return (cone_sphere_center_cos >= cone_sphere_angle_sum_cos);
}

bool culling_sphere_cylinder_isect(Sphere sphere, Cylinder cylinder)
{
  float distance_squared = len_squared(sphere.center.xy - cylinder.center.xy);
  return (distance_squared < sqr(cylinder.radius + sphere.radius));
}

bool culling_sphere_frustum_isect(Sphere sphere, Frustum frustum)
{
  if (dot(vec4(sphere.center, 1.0), frustum.planes[0]) > sphere.radius) {
    return false;
  }
  if (dot(vec4(sphere.center, 1.0), frustum.planes[1]) > sphere.radius) {
    return false;
  }
  if (dot(vec4(sphere.center, 1.0), frustum.planes[2]) > sphere.radius) {
    return false;
  }
  if (dot(vec4(sphere.center, 1.0), frustum.planes[3]) > sphere.radius) {
    return false;
  }
  return true;
}

bool culling_sphere_tile_isect(Sphere sphere, CullingTile tile)
{
  /* Culling in view space for precision and simplicity. */
  sphere.center = transform_point(ViewMatrix, sphere.center);
  bool isect;
  /* Test tile intersection using bounding cone or bounding cylinder.
   * This has less false positive cases when the sphere is large. */
  if (ProjectionMatrix[3][3] == 0.0) {
    isect = culling_sphere_cone_isect(sphere, tile.cone);
  }
  else {
    Cylinder cylinder = Cylinder(tile.cone.direction, tile.cone.angle_cos);
    isect = culling_sphere_cylinder_isect(sphere, cylinder);
  }
  /* Refine using frustum test. If the sphere is small it avoids intersection
   * with a neighbor tile. */
  if (isect) {
    isect = culling_sphere_frustum_isect(sphere, tile.frustum);
  }
  return isect;
}

/** \} */

/* ---------------------------------------------------------------------- */
/** \name Culling shapes extraction
 * \{ */

vec4 plane_from_quad(vec3 v0, vec3 v1, vec3 v2, vec3 v3)
{
  vec3 nor = normalize(cross(v2 - v1, v0 - v1) + cross(v0 - v3, v2 - v3));
  return vec4(nor, -dot(nor, v2));
}

/* Corners are expected to be in viewspace. */
Cone cone_from_quad(vec3 corners[8])
{
  for (int i = 0; i < 4; i++) {
    corners[i] = normalize(corners[i]);
  }
  vec3 center = normalize(corners[0] + corners[1] + corners[2] + corners[3]);

  vec4 corners_cos;
  for (int i = 0; i < 4; i++) {
    corners_cos[i] = dot(center, corners[i]);
  }
  return Cone(center, max_v4(corners_cos));
}

/* Corners are expected to be in viewspace. Returns Z-aligned bounding cylinder. */
Cone cylinder_from_quad(vec3 corners[8])
{
  vec3 center = (corners[0] + corners[1] + corners[2] + corners[3]) * 0.25;

  vec4 corners_dist;
  for (int i = 0; i < 4; i++) {
    corners_dist[i] = distance_squared(center, corners[i]);
  }
  /* Return a cone. Later converted to cylinder. */
  return Cone(center, sqrt(max_v4(corners_dist)));
}

vec2 tile_to_ndc(CullingData culling, vec2 tile_co, vec2 offset)
{
  /* Add a margin to prevent culling too much if the frustum becomes too much unstable. */
  tile_co += /* culling.tile_margin * */ offset;
  return tile_co * culling.tile_to_uv_fac * 2.0 - 1.0;
}

CullingTile culling_tile_get(CullingData culling, uvec2 tile_co)
{
  vec2 ftile = vec2(tile_co);
  /* Culling frustum corners for this tile. */
  vec3 corners[8];
  corners[0].xy = corners[4].xy = tile_to_ndc(culling, ftile, vec2(1, 1));
  corners[1].xy = corners[5].xy = tile_to_ndc(culling, ftile, vec2(1, 0));
  corners[2].xy = corners[6].xy = tile_to_ndc(culling, ftile, vec2(0, 0));
  corners[3].xy = corners[7].xy = tile_to_ndc(culling, ftile, vec2(0, 1));
  /* The corners depth only matter for precision. Use a mix of not so close to clip plane to
   * avoid small float imprecision if near clip is low. */
  corners[0].z = corners[1].z = corners[2].z = corners[3].z = -0.5;
  corners[4].z = corners[5].z = corners[6].z = corners[7].z = 0.1;

  for (int i = 0; i < 8; i++) {
    /* Culling in view space for precision. */
    corners[i] = project_point(ProjectionMatrixInverse, corners[i]);
  }

  bool is_persp = ProjectionMatrix[3][3] == 0.0;
  CullingTile tile;
  tile.cone = (is_persp) ? cone_from_quad(corners) : cylinder_from_quad(corners);
  tile.frustum.planes[0] = plane_from_quad(corners[0], corners[1], corners[5], corners[4]);
  tile.frustum.planes[1] = plane_from_quad(corners[1], corners[2], corners[6], corners[5]);
  tile.frustum.planes[2] = plane_from_quad(corners[2], corners[3], corners[7], corners[6]);
  tile.frustum.planes[3] = plane_from_quad(corners[3], corners[0], corners[4], corners[7]);
  return tile;
}

/** \} */
