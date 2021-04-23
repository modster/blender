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
 * The Original Code is Copyright (C) 2021 Blender Foundation
 * All rights reserved.
 */

/** \file
 * \ingroup spnode
 */

#include "BKE_context.h"

#include "BLI_math.h"
#include "BLI_rect.h"

#include "ED_screen.h"

#include "MEM_guardedalloc.h"

#include "PIL_time.h"

#include "RNA_access.h"
#include "RNA_define.h"

#include "UI_interface.h"
#include "UI_view2d.h"

#include "WM_api.h"
#include "WM_types.h"

/* -------------------------------------------------------------------- */
/** \name Edge Pan Operator Utilties
 * \{ */

bool UI_view2d_edge_pan_poll(bContext *C)
{
  ARegion *region = CTX_wm_region(C);

  /* check if there's a region in context to work with */
  if (region == NULL) {
    return false;
  }

  View2D *v2d = &region->v2d;

  /* check that 2d-view can pan */
  if ((v2d->keepofs & V2D_LOCKOFS_X) && (v2d->keepofs & V2D_LOCKOFS_Y)) {
    return false;
  }

  /* view can pan */
  return true;
}

void UI_view2d_edge_pan_init(struct bContext *C, struct View2DEdgePanData *vpd)
{
  UI_view2d_edge_pan_init_ex(
      C, vpd, /*region_pad=*/1.0f, /*speed_per_pixel=*/10.0f, /*delay=*/1.0f);
}

void UI_view2d_edge_pan_init_ex(
    bContext *C, View2DEdgePanData *vpd, float region_pad, float speed_per_pixel, float delay)
{
  if (!UI_view2d_edge_pan_poll(C)) {
    return;
  }

  /* set pointers to owners */
  vpd->screen = CTX_wm_screen(C);
  vpd->area = CTX_wm_area(C);
  vpd->region = CTX_wm_region(C);
  vpd->v2d = &vpd->region->v2d;

  vpd->region_pad = region_pad;
  vpd->speed_per_pixel = speed_per_pixel;
  vpd->delay = delay;

  /* calculate translation factor - based on size of view */
  const float winx = (float)(BLI_rcti_size_x(&vpd->region->winrct) + 1);
  const float winy = (float)(BLI_rcti_size_y(&vpd->region->winrct) + 1);
  vpd->facx = (BLI_rctf_size_x(&vpd->v2d->cur)) / winx;
  vpd->facy = (BLI_rctf_size_y(&vpd->v2d->cur)) / winy;

  UI_view2d_edge_pan_reset(vpd);
}

void UI_view2d_edge_pan_reset(View2DEdgePanData *vpd)
{
  vpd->edge_pan_start_time_x = 0.0;
  vpd->edge_pan_start_time_y = 0.0;
  vpd->edge_pan_last_time = PIL_check_seconds_timer();
}

void UI_view2d_edge_pan_apply(bContext *C, View2DEdgePanData *vpd, float dx, float dy)
{
  View2D *v2d = vpd->v2d;
  if (!v2d) {
    return;
  }

  /* calculate amount to move view by */
  dx *= vpd->facx;
  dy *= vpd->facy;

  /* only move view on an axis if change is allowed */
  if ((v2d->keepofs & V2D_LOCKOFS_X) == 0) {
    v2d->cur.xmin += dx;
    v2d->cur.xmax += dx;
  }
  if ((v2d->keepofs & V2D_LOCKOFS_Y) == 0) {
    v2d->cur.ymin += dy;
    v2d->cur.ymax += dy;
  }

  /* Inform v2d about changes after this operation. */
  UI_view2d_curRect_changed(C, v2d);

  /* don't rebuild full tree in outliner, since we're just changing our view */
  ED_region_tag_redraw_no_rebuild(vpd->region);

  /* request updates to be done... */
  WM_event_add_mousemove(CTX_wm_window(C));

  UI_view2d_sync(vpd->screen, vpd->area, v2d, V2D_LOCK_COPY);
}

void UI_view2d_edge_pan_operator_properties(wmOperatorType *ot)
{
  RNA_def_int(ot->srna,
              "outside_padding",
              0,
              0,
              100,
              "Outside Padding",
              "Padding around the region in UI units within which panning is activated (0 to "
              "disable boundary)",
              0,
              100);
}

/**
 * Reset the edge pan timers if the mouse isn't in the scroll zone and
 * start the timers when the mouse enters a scroll zone.
 */
static void edge_pan_manage_delay_timers(View2DEdgePanData *vpd,
                                         int pan_dir_x,
                                         int pan_dir_y,
                                         const double current_time)
{
  if (pan_dir_x == 0) {
    vpd->edge_pan_start_time_x = 0.0;
  }
  else if (vpd->edge_pan_start_time_x == 0.0) {
    vpd->edge_pan_start_time_x = current_time;
  }
  if (pan_dir_y == 0) {
    vpd->edge_pan_start_time_y = 0.0;
  }
  else if (vpd->edge_pan_start_time_y == 0.0) {
    vpd->edge_pan_start_time_y = current_time;
  }
}

/**
 * Used to calculate a "fade in" factor for edge panning to make the interaction feel smooth
 * and more purposeful.
 *
 * \note Assumes a domain_min of 0.0f.
 */
static float smootherstep(const float domain_max, float x)
{
  x = clamp_f(x / domain_max, 0.0, 1.0);
  return x * x * x * (x * (x * 6.0 - 15.0) + 10.0);
}

static float edge_pan_speed(View2DEdgePanData *vpd,
                            int event_loc,
                            bool x_dir,
                            const double current_time)
{
  ARegion *region = vpd->region;

  /* Find the distance from the start of the drag zone. */
  const int pad = vpd->region_pad * U.widget_unit;
  const int min = (x_dir ? region->winrct.xmin : region->winrct.ymin) + pad;
  const int max = (x_dir ? region->winrct.xmax : region->winrct.ymax) - pad;
  int distance = 0.0;
  if (event_loc > max) {
    distance = event_loc - max;
  }
  else if (event_loc < min) {
    distance = min - event_loc;
  }
  else {
    BLI_assert(!"Calculating speed outside of pan zones");
    return 0.0f;
  }

  /* Apply a fade in to the speed based on a start time delay. */
  const double start_time = x_dir ? vpd->edge_pan_start_time_x : vpd->edge_pan_start_time_y;
  const float delay_factor = smootherstep(vpd->delay, (float)(current_time - start_time));

  return distance * delay_factor * vpd->speed_per_pixel * (float)U.dpi_fac;
}

void UI_view2d_edge_pan_operator_apply(bContext *C,
                                       View2DEdgePanData *vpd,
                                       wmOperator *op,
                                       const wmEvent *event)
{
  ARegion *region = vpd->region;

  /* Only mousemove events matter here, ignore others. */
  if (event->type != MOUSEMOVE) {
    return;
  }

  const int outside_padding = RNA_int_get(op->ptr, "outside_padding") * UI_UNIT_X;
  rcti padding_rect;
  if (outside_padding != 0) {
    padding_rect = region->winrct;
    BLI_rcti_pad(&padding_rect, outside_padding, outside_padding);
  }

  int pan_dir_x = 0;
  int pan_dir_y = 0;
  if ((outside_padding == 0) || BLI_rcti_isect_pt(&padding_rect, event->x, event->y)) {
    const int pad = vpd->region_pad * U.widget_unit;
    /* Find whether the mouse is beyond X and Y edges. */
    if (event->x > region->winrct.xmax - pad) {
      pan_dir_x = 1;
    }
    else if (event->x < region->winrct.xmin + pad) {
      pan_dir_x = -1;
    }
    if (event->y > region->winrct.ymax - pad) {
      pan_dir_y = 1;
    }
    else if (event->y < region->winrct.ymin + pad) {
      pan_dir_y = -1;
    }
  }

  const double current_time = PIL_check_seconds_timer();
  edge_pan_manage_delay_timers(vpd, pan_dir_x, pan_dir_y, current_time);

  /* Calculate the delta since the last time the operator was called. */
  const float dtime = (float)(current_time - vpd->edge_pan_last_time);
  float dx = 0.0f, dy = 0.0f;
  if (pan_dir_x != 0) {
    const float speed = edge_pan_speed(vpd, event->x, true, current_time);
    dx = dtime * speed * (float)pan_dir_x;
  }
  if (pan_dir_y != 0) {
    const float speed = edge_pan_speed(vpd, event->y, false, current_time);
    dy = dtime * speed * (float)pan_dir_y;
  }
  vpd->edge_pan_last_time = current_time;

  /* Pan, clamping inside the regions's total bounds. */
  UI_view2d_edge_pan_apply(C, vpd, dx, dy);
}

/** \} */
