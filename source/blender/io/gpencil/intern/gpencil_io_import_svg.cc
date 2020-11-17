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
#include <iostream>
#include <list>
#include <string>

#include "MEM_guardedalloc.h"

#include "BKE_context.h"
#include "BKE_gpencil.h"
#include "BKE_gpencil_geom.h"
#include "BKE_main.h"
#include "BKE_material.h"

#include "BLI_blenlib.h"
#include "BLI_math.h"
#include "BLI_string.h"
#include "BLI_utildefines.h"

#include "DNA_gpencil_types.h"
#include "DNA_material_types.h"
#include "DNA_object_types.h"
#include "DNA_scene_types.h"
#include "DNA_screen_types.h"
#include "DNA_view3d_types.h"

#include "DEG_depsgraph.h"
#include "DEG_depsgraph_query.h"

#include "ED_gpencil.h"

#include "UI_view2d.h"

#include "ED_view3d.h"

#include "gpencil_io_import_svg.h"
#include "gpencil_io_importer.h"

#define NANOSVG_ALL_COLOR_KEYWORDS
#define NANOSVG_IMPLEMENTATION
#include "nanosvg/nanosvg.h"

namespace blender::io::gpencil {

/* Constructor. */
GpencilImporterSVG::GpencilImporterSVG(const char *filename,
                                       const struct GpencilImportParams *iparams)
    : GpencilImporter(iparams)
{
  set_filename(filename);
}

/* Destructor. */
GpencilImporterSVG::~GpencilImporterSVG(void)
{
  /* Nothing to do yet. */
}

bool GpencilImporterSVG::read(void)
{
  bool result = true;
  NSVGimage *svg_data = NULL;
  svg_data = nsvgParseFromFile(filename_, "px", 96.0f);
  if (svg_data == NULL) {
    std::cout << " Could not open SVG.\n ";
    return false;
  }

  /* Create grease pencil object. */
  if (params_.ob_target == NULL) {
    params_.ob_target = create_object();
    object_created_ = true;
  }
  if (params_.ob_target == NULL) {
    std::cout << "Unable to create new object.\n";
    return false;
  }
  bGPdata *gpd = (bGPdata *)params_.ob_target->data;

  /* Loop all shapes. */
  for (NSVGshape *shape = svg_data->shapes; shape; shape = shape->next) {
    /* Check if the layer exist and create if needed. */
    bGPDlayer *gpl = (bGPDlayer *)BLI_findstring(
        &gpd->layers, shape->id, offsetof(bGPDlayer, info));
    if (gpl == NULL) {
      BKE_gpencil_layer_addnew(gpd, shape->id, true);
    }
    /* Check frame. */
    bGPDframe *gpf = BKE_gpencil_layer_frame_get(gpl, cfra_, GP_GETFRAME_ADD_NEW);
    /* Create materials. */
    bool is_stroke = (bool)shape->stroke.type;
    bool is_fill = (bool)shape->fill.type;
    if ((!is_stroke) && (!is_fill)) {
      is_stroke = true;
    }

    /* Create_shape materials. */
    const char *const mat_names[] = {"Stroke", "Fill", "Stroke and Fill"};
    int index = 0;
    if ((is_stroke) && (is_fill)) {
      index = 2;
    }
    else if ((!is_stroke) && (is_fill)) {
      index = 1;
    }
    int32_t mat_index = create_material(mat_names[index], is_stroke, is_fill);

    /* Loop all paths to create the stroke data. */
    for (NSVGpath *path = shape->paths; path; path = path->next) {
      create_stroke(gpf, path, mat_index);
    }
  }

  /* Free memory. */
  nsvgDelete(svg_data);

  return result;
}

void GpencilImporterSVG::create_stroke(struct bGPDframe *gpf,
                                       struct NSVGpath *path,
                                       int32_t mat_index)
{
  bGPDstroke *gps = BKE_gpencil_stroke_new(mat_index, path->npts, 1.0f);
  BLI_addtail(&gpf->strokes, gps);
  gps->editcurve = BKE_gpencil_stroke_editcurve_new(path->npts);

  bGPDcurve_point *ptc = NULL;
  for (int i = 0; i < path->npts - 1; i += 3) {
  }
}

}  // namespace blender::io::gpencil
