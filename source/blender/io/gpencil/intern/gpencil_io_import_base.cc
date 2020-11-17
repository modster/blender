

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
#include "DNA_screen_types.h"

#include "UI_view2d.h"

#include "ED_view3d.h"

#include "DEG_depsgraph.h"
#include "DEG_depsgraph_query.h"

#include "gpencil_io_import_base.h"
#include "gpencil_io_importer.h"

#include "pugixml.hpp"

namespace blender::io::gpencil {

/* Constructor. */
GpencilImporter::GpencilImporter(const struct GpencilImportParams *iparams)
{
  params_.frame_target = iparams->frame_target;
  params_.ob_target = iparams->ob_target;
  params_.region = iparams->region;
  params_.v3d = iparams->v3d;
  params_.C = iparams->C;
  params_.mode = iparams->mode;
  params_.flag = iparams->flag;
  params_.stroke_sample = iparams->stroke_sample;

  /* Easy access data. */
  bmain = CTX_data_main(params_.C);
  depsgraph = CTX_data_depsgraph_pointer(params_.C);
  scene = CTX_data_scene(params_.C);
  rv3d = (RegionView3D *)params_.region->regiondata;

  gpd = NULL;
  gpl_cur_ = NULL;
  gpf_cur_ = NULL;
  gps_cur_ = NULL;
}

/**
 * Set output file input_text full path.
 * \param C: Context.
 * \param filename: Path of the file provided by dialog.
 */
void GpencilImporter::set_filename(const char *filename)
{
  BLI_strncpy(in_filename_, filename, FILE_MAX);
  BLI_path_abs(in_filename_, BKE_main_blendfile_path(bmain));
}

struct bGPDlayer *GpencilImporter::gpl_current_get(void)
{
  return gpl_cur_;
}

void GpencilImporter::gpl_current_set(struct bGPDlayer *gpl)
{
  gpl_cur_ = gpl;
}

struct bGPDframe *GpencilImporter::gpf_current_get(void)
{
  return gpf_cur_;
}

void GpencilImporter::gpf_current_set(struct bGPDframe *gpf)
{
  gpf_cur_ = gpf;
}
struct bGPDstroke *GpencilImporter::gps_current_get(void)
{
  return gps_cur_;
}

void GpencilImporter::gps_current_set(struct bGPDstroke *gps)
{
  gps_cur_ = gps;
}

}  // namespace blender::io::gpencil
