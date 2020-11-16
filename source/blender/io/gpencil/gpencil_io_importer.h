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

#ifdef __cplusplus
extern "C" {
#endif

struct ARegion;
struct bContext;
struct View3D;

struct GpencilImportParams {
  bContext *C;
  ARegion *region;
  View3D *v3d;
  /** Grease pencil object. */
  struct Object *obact;
  /** Import mode.  */
  uint16_t mode;
  /** Start frame.  */
  int32_t frame_start;
  /** End frame.  */
  int32_t frame_end;
  /* Current frame. */
  int32_t framenum;
  /** Flags. */
  uint32_t flag;
  /** Select mode */
  uint16_t select;
  /** Stroke sampling. */
  float stroke_sample;
};

/* GpencilImportParams->flag. */
typedef enum eGpencilImportParams_Flag {
  GP_IMPORT_DUMMY = (1 << 0),
} eGpencilImportParams_Flag;

typedef enum eGpencilImport_Modes {
  GP_IMPORT_FROM_SVG = 0,
  /* Add new import formats here. */
} eGpencilImport_Modes;

bool gpencil_io_import(const char *filename, struct GpencilImportParams *iparams);

#ifdef __cplusplus
}
#endif
