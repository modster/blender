/* SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

/** \file
 * \ingroup geo
 */

#ifdef __cplusplus
extern "C" {
#endif

struct Mesh;

typedef enum eRemeshBlocksMode {
  /* Blocks. */
  REMESH_BLOCKS_CENTROID = 0,
  /* Smooth. */
  REMESH_BLOCKS_MASS_POINT = 1,
  /* Smooth with sharp edges. */
  REMESH_BLOCKS_SHARP_FEATURES = 2,
} eRemeshBlocksMode;

struct Mesh *GEO_mesh_remesh_blocks(const struct Mesh *mesh,
                                    const char remesh_flag,
                                    const eRemeshBlocksMode remesh_mode,
                                    const float threshold,
                                    const int hermite_num,
                                    const float scale,
                                    const int depth);

#ifdef __cplusplus
}
#endif
