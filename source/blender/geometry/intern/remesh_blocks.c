/*
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

/** \file
 * \ingroup geo
 */

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"
#include "DNA_modifier_types.h"
#include "DNA_customdata_types.h"

#include "BKE_mesh.h"
#include "BKE_mesh_runtime.h"

#include "BLI_math_vector.h"
#include "BLI_threads.h"

#include "GEO_mesh_remesh_blocks.h" /* own include */

#include "MEM_guardedalloc.h"

#ifdef WITH_REMESH_DUALCON
#  include "dualcon.h"
#endif

static void init_dualcon_mesh(DualConInput *input, const Mesh *mesh)
{
  memset(input, 0, sizeof(DualConInput));

  input->co = (void *)mesh->mvert;
  input->co_stride = sizeof(MVert);
  input->totco = mesh->totvert;

  input->mloop = (void *)mesh->mloop;
  input->loop_stride = sizeof(MLoop);

  BKE_mesh_runtime_looptri_ensure(mesh);
  input->looptri = (void *)mesh->runtime.looptris.array;
  input->tri_stride = sizeof(MLoopTri);
  input->tottri = mesh->runtime.looptris.len;

  INIT_MINMAX(input->min, input->max);
  BKE_mesh_minmax(mesh, input->min, input->max);
}

/* Simple structure to hold the output: a CDDM and two counters to
 * keep track of the current elements. */
typedef struct {
  Mesh *mesh;
  int curvert, curface;
} DualConOutput;

/* Allocate and initialize a DualConOutput. */
static void *dualcon_alloc_output(int totvert, int totquad)
{
  DualConOutput *output;

  if (!(output = MEM_callocN(sizeof(DualConOutput), "DualConOutput"))) {
    return NULL;
  }

  output->mesh = BKE_mesh_new_nomain(totvert, 0, 0, 4 * totquad, totquad);
  return output;
}

static void dualcon_add_vert(void *output_v, const float co[3])
{
  DualConOutput *output = output_v;
  Mesh *mesh = output->mesh;

  BLI_assert(output->curvert < mesh->totvert);

  copy_v3_v3(mesh->mvert[output->curvert].co, co);
  output->curvert++;
}

static void dualcon_add_quad(void *output_v, const int vert_indices[4])
{
  DualConOutput *output = output_v;
  Mesh *mesh = output->mesh;
  MLoop *mloop;
  MPoly *cur_poly;
  int i;

  BLI_assert(output->curface < mesh->totpoly);

  mloop = mesh->mloop;
  cur_poly = &mesh->mpoly[output->curface];

  cur_poly->loopstart = output->curface * 4;
  cur_poly->totloop = 4;
  for (i = 0; i < 4; i++) {
    mloop[output->curface * 4 + i].v = vert_indices[i];
  }

  output->curface++;
}

Mesh *GEO_mesh_remesh_blocks(const Mesh *mesh,
                             const char remesh_flag,
                             const eRemeshBlocksMode remesh_mode,
                             const float threshold,
                             const int hermite_num,
                             const float scale,
                             const int depth)
{
#ifdef WITH_REMESH_DUALCON

  DualConOutput *output;
  DualConInput input;
  Mesh *result;
  DualConFlags flags = 0;
  DualConMode mode = 0;

  /* Dualcon modes. */
  init_dualcon_mesh(&input, mesh);

  if (remesh_flag & MOD_REMESH_FLOOD_FILL) {
    flags |= DUALCON_FLOOD_FILL;
  }

  switch (remesh_mode) {
    case REMESH_BLOCKS_CENTROID:
      mode = DUALCON_CENTROID;
      break;
    case REMESH_BLOCKS_MASS_POINT:
      mode = DUALCON_MASS_POINT;
      break;
    case REMESH_BLOCKS_SHARP_FEATURES:
      mode = DUALCON_SHARP_FEATURES;
      break;
  }
  /* TODO(jbakker): Dualcon crashes when run in parallel. Could be related to incorrect
   * input data or that the library isn't thread safe.
   * This was identified when changing the task isolation's during T76553. */
  static ThreadMutex dualcon_mutex = BLI_MUTEX_INITIALIZER;
  BLI_mutex_lock(&dualcon_mutex);
  output = dualcon(&input,
                   dualcon_alloc_output,
                   dualcon_add_vert,
                   dualcon_add_quad,
                   flags,
                   mode,
                   threshold,
                   hermite_num,
                   scale,
                   depth);
  BLI_mutex_unlock(&dualcon_mutex);

  result = output->mesh;
  MEM_freeN(output);

  return result;
#else
  return BKE_mesh_new_nomain(0,0,0,0,0);
#endif
}
