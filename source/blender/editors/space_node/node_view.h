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
 * The Original Code is Copyright (C) 2021 Blender Foundation, Lukas Toenne
 * All rights reserved.
 */

/** \file
 * \ingroup spnode
 */

#pragma once

struct bContext;
struct bScreen;
struct ScrArea;
struct ARegion;
struct View2D;
struct wmEvent;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Custom-data for view panning operators.
 */
typedef struct NodeViewPanData {
  /** screen where view pan was initiated */
  bScreen *screen;
  /** area where view pan was initiated */
  ScrArea *area;
  /** region where view pan was initiated */
  ARegion *region;
  /** view2d we're operating in */
  View2D *v2d;

  /** amount to move view relative to zoom */
  float facx, facy;

  /* View2D Edge Panning */
  double edge_pan_last_time;
  double edge_pan_start_time_x, edge_pan_start_time_y;
} NodeViewPanData;

bool node_edge_pan_poll(struct bContext *C);

/* Initialize panning customdata. */
void node_edge_pan_init(struct bContext *C, NodeViewPanData *vpd);

/* Initialize timers when operator starts. */
void node_edge_pan_start(struct bContext *C, NodeViewPanData *vpd, struct wmOperator *op);

/* apply transform to view (i.e. adjust 'cur' rect). */
void node_edge_pan_apply_ex(struct bContext *C, NodeViewPanData *vpd, float dx, float dy);

/* Define operator properties needed for view panning. */
void node_edge_pan_properties(struct wmOperatorType *ot);

/* Apply to view using operator properties. */
void node_edge_pan_apply_op(struct bContext *C,
                            NodeViewPanData *vpd,
                            struct wmOperator *op,
                            const struct wmEvent *event);

#ifdef __cplusplus
}
#endif
