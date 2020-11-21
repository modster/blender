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

/* Paper Size: A4, Letter. */
static const float gpencil_export_paper_sizes[2] = {3508, 2480};

struct GpencilExportParams {
  bContext *C;
  ARegion *region;
  View3D *v3d;
  /** Grease pencil object. */
  struct Object *obact;
  /** Export mode.  */
  uint16_t mode;
  /** Start frame.  */
  int32_t frame_start;
  /** End frame.  */
  int32_t frame_end;
  /* Current frame. */
  int32_t framenum;
  /** Flags. */
  uint32_t flag;
  /** Select mode. */
  uint16_t select;
  /** Frame type. */
  uint16_t frame_type;
  /** File subfix. */
  char file_subfix[10];
  /** Stroke sampling. */
  float stroke_sample;
  /** Paper size in pixels. */
  float paper_size[2];
};

/* GpencilExportParams->flag. */
typedef enum eGpencilExportParams_Flag {
  /* Export Filled strokes. */
  GP_EXPORT_FILL = (1 << 0),
  /* Export normalized thickness. */
  GP_EXPORT_NORM_THICKNESS = (1 << 1),
  /* Clip camera area. */
  GP_EXPORT_CLIP_CAMERA = (1 << 2),
} eGpencilExportParams_Flag;

typedef enum eGpencilExport_Modes {
  GP_EXPORT_TO_SVG = 0,
  GP_EXPORT_TO_PDF = 1,
  /* Add new export formats here. */
} eGpencilExport_Modes;

/* Object to be exported. */
typedef enum eGpencilExportSelect {
  GP_EXPORT_ACTIVE = 0,
  GP_EXPORT_SELECTED = 1,
  GP_EXPORT_VISIBLE = 2,
} eGpencilExportSelect;

/* Framerange to be exported. */
typedef enum eGpencilExportFrame {
  GP_EXPORT_FRAME_ACTIVE = 0,
  GP_EXPORT_FRAME_SELECTED = 1,
} eGpencilExportFrame;

bool gpencil_io_export(const char *filename, struct GpencilExportParams *iparams);

#ifdef __cplusplus
}
#endif
