/* SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup DNA
 */

#pragma once

#include "DNA_ID.h"
#include "DNA_curves_types.h"
#include "DNA_customdata_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
namespace blender::bke {
class GPDataRuntime;
}  // namespace blender::bke
using GPDataRuntimeHandle = blender::bke::GPDataRuntime;
#else
typedef struct GPDataRuntimeHandle GPDataRuntimeHandle;
#endif

typedef struct GPLayerGroup {
  struct GPLayerGroup *children;
  int children_size;

  int *layer_indices;
  int layer_indices_size;

  char name[128];

  /* ... */
} GPLayerGroup;

typedef struct GPLayer {
  char name[128];

  int flag;

  /* ... */
} GPLayer;

typedef struct GPFrame {
  CurvesGeometry strokes;

  int flag;

  int start;
  int end;

  /* ... */
} GPFrame;

typedef struct GPData {
  GPFrame *frames_array;
  int frames_size;
  CustomData frame_data;
  int active_frame_index;

  GPLayer *layers_array;
  int layers_size;
  int active_layer_index;

  GPLayerGroup *default_group;

  GPDataRuntimeHandle *runtime;
} GPData;

/* This is the ID_GP structure that holds all the */
typedef struct GreasePencil {
  ID id;
  /* Animation data (must be immediately after id). */
  struct AnimData *adt;

  /* Pointer to the actual data-block containing the frames, layers and layer groups. */
  GPData *grease_pencil_data;

  int flag;

  /** Materials array. */
  struct Material **mat;
  /** Total materials. */
  short totcol;

  /* ... */
} GreasePencil;

#ifdef __cplusplus
}
#endif