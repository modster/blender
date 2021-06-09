#include "DNA_object_types.h"

#ifdef __cplusplus
extern "C" {
#endif

struct Mesh;

typedef struct SolidifyData {
  /** Name of vertex group to use, MAX_VGROUP_NAME. */
  const Object *object;
  char defgrp_name[64];
  char shell_defgrp_name[64];
  char rim_defgrp_name[64];
  /** New surface offset level. */
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
  char mode;

  /** Variables for #MOD_SOLIDIFY_MODE_NONMANIFOLD. */
  char nonmanifold_offset_mode;
  char nonmanifold_boundary_mode;

  float crease_inner;
  float crease_outer;
  float crease_rim;
  int flag;
  short mat_ofs;
  short mat_ofs_rim;

  float merge_tolerance;
  float bevel_convex;
  float *selection;
} SolidifyData;

Mesh *solidify_extrude(const SolidifyData *solidify_data, Mesh *mesh);
Mesh *solidify_nonmanifold(const SolidifyData *solidify_data, Mesh *mesh);

#ifdef __cplusplus
}
#endif
