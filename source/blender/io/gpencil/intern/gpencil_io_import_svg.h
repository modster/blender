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
#pragma once

/** \file
 * \ingroup bgpencil
 */
#include "gpencil_io_import_base.h"

struct GpencilImportParams;
struct NSVGpath;
struct bGPdata;
struct bGPDframe;

#define SVG_IMPORTER_NAME "SVG Import for Grease Pencil"
#define SVG_IMPORTER_VERSION "v1.0"

namespace blender::io::gpencil {

class GpencilImporterSVG : public GpencilImporter {

 public:
  GpencilImporterSVG(const char *filename, const struct GpencilImportParams *iparams);
  ~GpencilImporterSVG(void);

  bool read(void);

 protected:
 private:
  void create_stroke(struct bGPdata *gpd,
                     struct bGPDframe *gpf,
                     struct NSVGpath *path,
                     int32_t mat_index);
};

}  // namespace blender::io::gpencil
