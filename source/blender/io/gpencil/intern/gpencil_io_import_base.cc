

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
 * The Original Code is Copyright (C) 2020 Blender Foundation
 * All rights reserved.
 */

/** \file
 * \ingroup bgpencil
 */
#include <algorithm>
#include <cctype>
#include <iostream>
#include <string>

#include "BKE_context.h"
#include "BKE_gpencil.h"
#include "BKE_gpencil_geom.h"
#include "BKE_layer.h"
#include "BKE_main.h"
#include "BKE_material.h"

#include "BLI_blenlib.h"
#include "BLI_math.h"
#include "BLI_utildefines.h"

#include "DNA_gpencil_types.h"
#include "DNA_material_types.h"
#include "DNA_object_types.h"
#include "DNA_scene_types.h"
#include "DNA_screen_types.h"

#include "UI_view2d.h"

#include "ED_gpencil.h"
#include "ED_view3d.h"

#include "DEG_depsgraph.h"
#include "DEG_depsgraph_query.h"

#include "gpencil_io.h"
#include "gpencil_io_import_base.h"

#include "pugixml.hpp"

namespace blender::io::gpencil {

/* Constructor. */
GpencilImporter::GpencilImporter(const struct GpencilIOParams *iparams) : GpencilIO(iparams)
{
  /* Nothing to do yet */
}

Object *GpencilImporter::create_object(void)
{
  const float *cur = scene_->cursor.location;
  ushort local_view_bits = (params_.v3d && params_.v3d->localvd) ? params_.v3d->local_view_uuid :
                                                                   (ushort)0;
  Object *ob_gpencil = ED_gpencil_add_object(params_.C, cur, local_view_bits);

  return ob_gpencil;
}

int32_t GpencilImporter::create_material(const char *name, const bool stroke, const bool fill)
{
  const float default_stroke_color[4] = {0.0f, 0.0f, 0.0f, 1.0f};
  const float default_fill_color[4] = {0.5f, 0.5f, 0.5f, 1.0f};
  int32_t mat_index = BKE_gpencil_material_find_index_by_name_prefix(params_.ob, name);
  /* Stroke and Fill material. */
  if (mat_index == -1) {
    int32_t new_idx;
    Material *mat_gp = BKE_gpencil_object_material_new(bmain_, params_.ob, name, &new_idx);
    MaterialGPencilStyle *gp_style = mat_gp->gp_style;
    gp_style->flag &= ~GP_MATERIAL_STROKE_SHOW;
    gp_style->flag &= ~GP_MATERIAL_FILL_SHOW;

    copy_v4_v4(gp_style->stroke_rgba, default_stroke_color);
    copy_v4_v4(gp_style->fill_rgba, default_fill_color);
    if (stroke) {
      gp_style->flag |= GP_MATERIAL_STROKE_SHOW;
    }
    if (fill) {
      gp_style->flag |= GP_MATERIAL_FILL_SHOW;
    }
    mat_index = params_.ob->totcol - 1;
  }

  return mat_index;
}

}  // namespace blender::io::gpencil
