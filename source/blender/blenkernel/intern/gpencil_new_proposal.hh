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
  /**
   * An array of GPLayerGroup's. A layer group can have N >= 0 number of layer group children.
   */
  struct GPLayerGroup *children;
  int children_size;

  /**
   * An array of indices to the layers in GPData.layers_array. These are the layers contained in
   * the group.
   */
  int *layer_indices;
  int layer_indices_size;

  /**
   * The name of the layer group.
   */
  char name[128];

  /* ... */
} GPLayerGroup;

typedef struct GPLayer {
  /**
   * The name of the layer.
   */
  char name[128];

  /**
   * The layer flag.
   */
  int flag;

  /* ... */
} GPLayer;

typedef struct GPFrame {
  /**
   * The curves in this frame. Each individual curve is a single stroke. The CurvesGeometry
   * structure also stores attributes on the strokes and points.
   */
  CurvesGeometry strokes;

  /**
   * The frame flag.
   */
  int flag;

  /**
   * The index of the layer in GPData.layers_array that this frame is in.
   */
  int layer_index;

  /**
   * The start and end frame in the scene that the grease pencil frame is displayed.
   */
  int start;
  int end;

  /* ... */
} GPFrame;

typedef struct GPData {
  /**
   * The array of grease pencil frames. This is kept in chronological order (tiebreaks for two
   * frames on different layers are resloved by the order of the layers).
   */
  GPFrame *frames_array;
  int frames_size;

  /**
   * All attributes stored on the frames.
   */
  CustomData frame_data;

  /**
   * The array of grease pencil layers.
   */
  GPLayer *layers_array;
  int layers_size;

  /**
   * The index of the active layer in the GPData.layers_array.
   */
  int active_layer_index;

  /**
   * The root layer group. This must not be nullptr.
   */
  GPLayerGroup *default_group;

  /**
   * The runtime data.
   */
  GPDataRuntimeHandle *runtime;
} GPData;

/* This is the ID_GP structure that holds all the information at the object data level. */
typedef struct GreasePencil {
  ID id;
  /* Animation data (must be immediately after id). */
  struct AnimData *adt;

  /* Pointer to the actual data-block containing the frames, layers and layer groups. */
  GPData *grease_pencil_data;

  /* GreasePencil flag. */
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