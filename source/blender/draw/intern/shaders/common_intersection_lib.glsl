
#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_geom_lib.glsl)

/** Require/include common_debug_lib.glsl before this file to debug draw intersections volumes. */

#if defined(DEBUG_DRAW_ISECT) && !defined(DEBUG_DRAW)
#  error "You must include common_debug_lib.glsl before this file to enabled debug draw"
#endif

/* ---------------------------------------------------------------------- */
/** \name Plane extraction functions.
 * \{ */

/** \a v1 and \a v2 are vectors on the plane. \a p is a point on the plane. */
vec4 plane_setup(vec3 p, vec3 v1, vec3 v2)
{
  vec3 normal_to_plane = normalize(cross(v1, v2));
  return vec4(normal_to_plane, -dot(normal_to_plane, p));
}

void planes_setup(Pyramid shape, out vec4 planes[5])
{
  vec3 A1 = shape.corners[1] - shape.corners[0];
  vec3 A2 = shape.corners[2] - shape.corners[0];
  vec3 A3 = shape.corners[3] - shape.corners[0];
  vec3 A4 = shape.corners[4] - shape.corners[0];
  vec3 S1 = shape.corners[4] - shape.corners[1];
  vec3 S2 = shape.corners[2] - shape.corners[1];

  planes[0] = plane_setup(shape.corners[0], A2, A1);
  planes[1] = plane_setup(shape.corners[0], A3, A2);
  planes[2] = plane_setup(shape.corners[0], A4, A3);
  planes[3] = plane_setup(shape.corners[0], A1, A4);
  planes[4] = plane_setup(shape.corners[1], S2, S1);
}

void planes_setup(Box shape, out vec4 planes[6])
{
  vec3 A1 = shape.corners[1] - shape.corners[0];
  vec3 A3 = shape.corners[3] - shape.corners[0];
  vec3 A4 = shape.corners[4] - shape.corners[0];

  planes[0] = plane_setup(shape.corners[0], A1, A3);
  planes[1] = plane_setup(shape.corners[0], A3, A4);
  planes[2] = plane_setup(shape.corners[0], A4, A1);
  planes[3] = vec4(-planes[0].xyz, -dot(-planes[0].xyz, shape.corners[6]));
  planes[4] = vec4(-planes[1].xyz, -dot(-planes[1].xyz, shape.corners[6]));
  planes[5] = vec4(-planes[2].xyz, -dot(-planes[2].xyz, shape.corners[6]));
}

/** \} */

/* ---------------------------------------------------------------------- */
/** \name Intersection functions.
 * \{ */

#define TEST_ENABLED 1
#define FALSE_POSITIVE_REJECTION 1

bool intersect_view(Pyramid pyramid)
{
  /**
   * Frustum vs. Pyramid test from
   * https://www.yosoygames.com.ar/wp/2016/12/frustum-vs-pyramid-intersection-also-frustum-vs-frustum/
   */
  bool intersects = true;

#if TEST_ENABLED
  /* Do Pyramid vertices vs Frustum planes. */
  for (int p = 0; p < 6 && intersects; ++p) {
    bool is_any_vertex_on_positive_side = false;
    for (int v = 0; v < 5 && !is_any_vertex_on_positive_side; ++v) {
      if (dot(frustum_planes[p], vec4(pyramid.corners[v], 1.0)) > 0.0) {
        is_any_vertex_on_positive_side = true;
      }
    }
    if (!is_any_vertex_on_positive_side) {
      intersects = false;
    }
  }
#endif

#if TEST_ENABLED && FALSE_POSITIVE_REJECTION
  if (intersects) {
    vec4 pyramid_planes[5];
    planes_setup(pyramid, pyramid_planes);
    /* Now do Frustum vertices vs Pyramid planes. */
    for (int p = 0; p < 5 && intersects; ++p) {
      bool is_any_vertex_on_positive_side = false;
      for (int v = 0; v < 8 && !is_any_vertex_on_positive_side; ++v) {
        if (dot(pyramid_planes[p], vec4(frustum_corners[v], 1.0)) > 0.0) {
          is_any_vertex_on_positive_side = true;
        }
      }
      if (!is_any_vertex_on_positive_side) {
        intersects = false;
      }
    }
  }
#endif

#if defined(DEBUG_DRAW) && defined(DEBUG_DRAW_ISECT)
  drw_debug(pyramid, intersects ? vec4(0, 1, 0, 1) : vec4(1, 0, 0, 1));
#endif
  return intersects;
}

bool intersect_view(Box box)
{
  bool intersects = true;

#if TEST_ENABLED
  /* Do Box vertices vs Frustum planes. */
  for (int p = 0; p < 6 && intersects; ++p) {
    bool is_any_vertex_on_positive_side = false;
    for (int v = 0; v < 8 && !is_any_vertex_on_positive_side; ++v) {
      if (dot(frustum_planes[p], vec4(box.corners[v], 1.0)) > 0.0) {
        is_any_vertex_on_positive_side = true;
      }
    }
    if (!is_any_vertex_on_positive_side) {
      intersects = false;
    }
  }
#endif

#if TEST_ENABLED && FALSE_POSITIVE_REJECTION
  if (intersects) {
    vec4 box_planes[6];
    planes_setup(box, box_planes);
    /* Now do Frustum vertices vs Box planes. */
    for (int p = 0; p < 6 && intersects; ++p) {
      bool is_any_vertex_on_positive_side = false;
      for (int v = 0; v < 8 && !is_any_vertex_on_positive_side; ++v) {
        if (dot(box_planes[p], vec4(frustum_corners[v], 1.0)) > 0.0) {
          is_any_vertex_on_positive_side = true;
        }
      }
      if (!is_any_vertex_on_positive_side) {
        intersects = false;
      }
    }
  }
#endif

#if defined(DEBUG_DRAW) && defined(DEBUG_DRAW_ISECT)
  if (intersects) {
    drw_debug(box, vec4(0, 1, 0, 1));
  }
#endif
  return intersects;
}

bool intersect(Pyramid pyramid, Box box)
{
  bool intersects = true;

  vec4 box_planes[6];
  planes_setup(box, box_planes);
  /* Do Pyramid vertices vs Box planes. */
  for (int p = 0; p < 6 && intersects; ++p) {
    bool is_any_vertex_on_positive_side = false;
    for (int v = 0; v < 5 && !is_any_vertex_on_positive_side; ++v) {
      if (dot(box_planes[p], vec4(pyramid.corners[v], 1.0)) > 0.0) {
        is_any_vertex_on_positive_side = true;
      }
    }
    if (!is_any_vertex_on_positive_side) {
      intersects = false;
    }
  }

  if (intersects) {
    vec4 pyramid_planes[5];
    planes_setup(pyramid, pyramid_planes);
    /* Now do Box vertices vs Pyramid planes. */
    for (int p = 0; p < 5 && intersects; ++p) {
      bool is_any_vertex_on_positive_side = false;
      for (int v = 0; v < 8 && !is_any_vertex_on_positive_side; ++v) {
        if (dot(pyramid_planes[p], vec4(box.corners[v], 1.0)) > 0.0) {
          is_any_vertex_on_positive_side = true;
        }
      }
      if (!is_any_vertex_on_positive_side) {
        intersects = false;
      }
    }
  }
  return intersects;
}

#undef TEST_ENABLED
#undef FALSE_POSITIVE_REJECTION

/** \} */
