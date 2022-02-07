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
 *
 * The Original Code is Copyright (C) 2019 Blender Foundation.
 * All rights reserved.
 */

/** \file
 * \ingroup depsgraph
 */

#include "intern/eval/deg_eval_runtime_backup_gpencil.h"
#include "intern/depsgraph.h"

#include "BKE_gpencil.h"
#include "BKE_gpencil_update_cache.h"

#include "DNA_gpencil_types.h"

namespace blender::deg {

GPencilBackup::GPencilBackup(const Depsgraph *depsgraph) : depsgraph(depsgraph)
{
}

void GPencilBackup::init_from_gpencil(bGPdata *gpd)
{
}

void GPencilBackup::restore_to_gpencil(bGPdata *gpd)
{
  bGPdata *gpd_orig = reinterpret_cast<bGPdata *>(gpd->id.orig_id);
  if (depsgraph->is_active) {
    BKE_gpencil_free_update_cache(gpd_orig);
    gpd->runtime.update_cache = NULL;
  }
  BKE_gpencil_data_update_orig_pointers(gpd_orig, gpd);
}

}  // namespace blender::deg
