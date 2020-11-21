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
#include <list>
#include <string>

#include "BLI_path_util.h"

#include "DNA_defs.h"

#include "gpencil_io_importer.h"

struct ARegion;
struct Depsgraph;
struct Main;
struct Object;
struct RegionView3D;
struct Scene;

struct bGPdata;
struct bGPDlayer;
struct bGPDframe;
struct bGPDstroke;

namespace blender::io::gpencil {

class GpencilImporter {

 public:
  GpencilImporter(const struct GpencilImportParams *iparams);
  virtual bool read(void) = 0;

  void set_frame_number(int value);
  struct Object *create_object(void);
  int32_t create_material(const char *name, const bool stroke, const bool fill);

 protected:
  GpencilImportParams params_;

  bool invert_axis_[2];
  float diff_mat_[4][4];
  char filename_[FILE_MAX];

  /* Data for easy access. */
  struct Depsgraph *depsgraph_;
  struct bGPdata *gpd_;
  struct Main *bmain_;
  struct Scene *scene_;
  struct RegionView3D *rv3d_;

  int cfra_;
  bool object_created_;

  struct bGPDlayer *gpl_current_get(void);
  struct bGPDframe *gpf_current_get(void);
  struct bGPDstroke *gps_current_get(void);
  void gpl_current_set(struct bGPDlayer *gpl);
  void gpf_current_set(struct bGPDframe *gpf);
  void gps_current_set(struct bGPDstroke *gps);

  void set_filename(const char *filename);

 private:
  struct bGPDlayer *gpl_cur_;
  struct bGPDframe *gpf_cur_;
  struct bGPDstroke *gps_cur_;
};

}  // namespace blender::io::gpencil
