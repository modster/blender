#include "DNA_object_types.h"

#ifdef __cplusplus
extern "C" {
#endif

struct Mesh;

typedef struct SolidifyData {
  const Object *object;
  /** New surface offset level. (Thickness) */
  float offset;
  /** Midpoint of the offset. */
  float offset_fac;
  /**
   * Factor for the minimum weight to use when vertex-groups are used,
   * avoids 0.0 weights giving duplicate geometry.
   */
  float offset_fac_vg;
  /** Clamp offset based on surrounding geometry. */
  float offset_clamp;

  /** Variables for #MOD_SOLIDIFY_MODE_NONMANIFOLD. */
  char nonmanifold_offset_mode;
  char nonmanifold_boundary_mode;

  int flag;

  float merge_tolerance;
  float bevel_convex;
  float *distance;
} SolidifyData;

struct Mesh *solidify_nonmanifold(const SolidifyData *solidify_data,
                           struct Mesh *mesh,
                           bool **r_shell_verts,
                           bool **r_rim_verts,
                           bool **r_shell_faces,
                           bool **r_rim_faces);

#ifdef __cplusplus
}
#endif
