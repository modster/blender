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
 * The Original Code is Copyright (C) 2008 Blender Foundation.
 * All rights reserved.
 */

/** \file
 * \ingroup spnode
 */

#include "DNA_node_types.h"

#include "BLI_math.h"
#include "BLI_rect.h"
#include "BLI_utildefines.h"

#include "BKE_context.h"
#include "BKE_image.h"
#include "BKE_main.h"
#include "BKE_node.h"
#include "BKE_screen.h"

#include "ED_image.h"
#include "ED_node.h" /* own include */
#include "ED_screen.h"
#include "ED_space_api.h"

#include "RNA_access.h"
#include "RNA_define.h"

#include "WM_api.h"
#include "WM_types.h"

#include "UI_view2d.h"

#include "MEM_guardedalloc.h"

#include "IMB_colormanagement.h"
#include "IMB_imbuf.h"
#include "IMB_imbuf_types.h"

#include "PIL_time.h"

#include "node_intern.h" /* own include */
#include "node_view.h" /* own include */

 /* -------------------------------------------------------------------- */
/** \name View All Operator
 * \{ */

int space_node_view_flag(
    bContext *C, SpaceNode *snode, ARegion *region, const int node_flag, const int smooth_viewtx)
{
  bNode *node;
  rctf cur_new;
  float oldwidth, oldheight, width, height;
  float oldasp, asp;
  int tot = 0;
  bool has_frame = false;

  oldwidth = BLI_rctf_size_x(&region->v2d.cur);
  oldheight = BLI_rctf_size_y(&region->v2d.cur);

  oldasp = oldwidth / oldheight;

  BLI_rctf_init_minmax(&cur_new);

  if (snode->edittree) {
    for (node = snode->edittree->nodes.first; node; node = node->next) {
      if ((node->flag & node_flag) == node_flag) {
        BLI_rctf_union(&cur_new, &node->totr);
        tot++;

        if (node->type == NODE_FRAME) {
          has_frame = true;
        }
      }
    }
  }

  if (tot) {
    width = BLI_rctf_size_x(&cur_new);
    height = BLI_rctf_size_y(&cur_new);
    asp = width / height;

    /* for single non-frame nodes, don't zoom in, just pan view,
     * but do allow zooming out, this allows for big nodes to be zoomed out */
    if ((tot == 1) && (has_frame == false) && ((oldwidth * oldheight) > (width * height))) {
      /* center, don't zoom */
      BLI_rctf_resize(&cur_new, oldwidth, oldheight);
    }
    else {
      if (oldasp < asp) {
        const float height_new = width / oldasp;
        cur_new.ymin = cur_new.ymin - height_new / 2.0f;
        cur_new.ymax = cur_new.ymax + height_new / 2.0f;
      }
      else {
        const float width_new = height * oldasp;
        cur_new.xmin = cur_new.xmin - width_new / 2.0f;
        cur_new.xmax = cur_new.xmax + width_new / 2.0f;
      }

      /* add some padding */
      BLI_rctf_scale(&cur_new, 1.1f);
    }

    UI_view2d_smooth_view(C, region, &cur_new, smooth_viewtx);
  }

  return (tot != 0);
}

static int node_view_all_exec(bContext *C, wmOperator *op)
{
  ARegion *region = CTX_wm_region(C);
  SpaceNode *snode = CTX_wm_space_node(C);
  const int smooth_viewtx = WM_operator_smooth_viewtx_get(op);

  /* is this really needed? */
  snode->xof = 0;
  snode->yof = 0;

  if (space_node_view_flag(C, snode, region, 0, smooth_viewtx)) {
    return OPERATOR_FINISHED;
  }
  return OPERATOR_CANCELLED;
}

void NODE_OT_view_all(wmOperatorType *ot)
{
  /* identifiers */
  ot->name = "Frame All";
  ot->idname = "NODE_OT_view_all";
  ot->description = "Resize view so you can see all nodes";

  /* api callbacks */
  ot->exec = node_view_all_exec;
  ot->poll = ED_operator_node_active;

  /* flags */
  ot->flag = 0;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name View Selected Operator
 * \{ */

static int node_view_selected_exec(bContext *C, wmOperator *op)
{
  ARegion *region = CTX_wm_region(C);
  SpaceNode *snode = CTX_wm_space_node(C);
  const int smooth_viewtx = WM_operator_smooth_viewtx_get(op);

  if (space_node_view_flag(C, snode, region, NODE_SELECT, smooth_viewtx)) {
    return OPERATOR_FINISHED;
  }
  return OPERATOR_CANCELLED;
}

void NODE_OT_view_selected(wmOperatorType *ot)
{
  /* identifiers */
  ot->name = "Frame Selected";
  ot->idname = "NODE_OT_view_selected";
  ot->description = "Resize view so you can see selected nodes";

  /* api callbacks */
  ot->exec = node_view_selected_exec;
  ot->poll = ED_operator_node_active;

  /* flags */
  ot->flag = 0;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Background Image Operators
 * \{ */

typedef struct NodeViewMove {
  int mvalo[2];
  int xmin, ymin, xmax, ymax;
} NodeViewMove;

static int snode_bg_viewmove_modal(bContext *C, wmOperator *op, const wmEvent *event)
{
  SpaceNode *snode = CTX_wm_space_node(C);
  ARegion *region = CTX_wm_region(C);
  NodeViewMove *nvm = op->customdata;

  switch (event->type) {
    case MOUSEMOVE:

      snode->xof -= (nvm->mvalo[0] - event->mval[0]);
      snode->yof -= (nvm->mvalo[1] - event->mval[1]);
      nvm->mvalo[0] = event->mval[0];
      nvm->mvalo[1] = event->mval[1];

      /* prevent dragging image outside of the window and losing it! */
      CLAMP(snode->xof, nvm->xmin, nvm->xmax);
      CLAMP(snode->yof, nvm->ymin, nvm->ymax);

      ED_region_tag_redraw(region);
      WM_main_add_notifier(NC_NODE | ND_DISPLAY, NULL);
      WM_main_add_notifier(NC_SPACE | ND_SPACE_NODE_VIEW, NULL);

      break;

    case LEFTMOUSE:
    case MIDDLEMOUSE:
    case RIGHTMOUSE:
      if (event->val == KM_RELEASE) {
        MEM_freeN(nvm);
        op->customdata = NULL;
        return OPERATOR_FINISHED;
      }
      break;
  }

  return OPERATOR_RUNNING_MODAL;
}

static int snode_bg_viewmove_invoke(bContext *C, wmOperator *op, const wmEvent *event)
{
  Main *bmain = CTX_data_main(C);
  SpaceNode *snode = CTX_wm_space_node(C);
  ARegion *region = CTX_wm_region(C);
  NodeViewMove *nvm;
  Image *ima;
  ImBuf *ibuf;
  const float pad = 32.0f; /* better be bigger than scrollbars */

  void *lock;

  ima = BKE_image_ensure_viewer(bmain, IMA_TYPE_COMPOSITE, "Viewer Node");
  ibuf = BKE_image_acquire_ibuf(ima, NULL, &lock);

  if (ibuf == NULL) {
    BKE_image_release_ibuf(ima, ibuf, lock);
    return OPERATOR_CANCELLED;
  }

  nvm = MEM_callocN(sizeof(NodeViewMove), "NodeViewMove struct");
  op->customdata = nvm;
  nvm->mvalo[0] = event->mval[0];
  nvm->mvalo[1] = event->mval[1];

  nvm->xmin = -(region->winx / 2) - (ibuf->x * (0.5f * snode->zoom)) + pad;
  nvm->xmax = (region->winx / 2) + (ibuf->x * (0.5f * snode->zoom)) - pad;
  nvm->ymin = -(region->winy / 2) - (ibuf->y * (0.5f * snode->zoom)) + pad;
  nvm->ymax = (region->winy / 2) + (ibuf->y * (0.5f * snode->zoom)) - pad;

  BKE_image_release_ibuf(ima, ibuf, lock);

  /* add modal handler */
  WM_event_add_modal_handler(C, op);

  return OPERATOR_RUNNING_MODAL;
}

static void snode_bg_viewmove_cancel(bContext *UNUSED(C), wmOperator *op)
{
  MEM_freeN(op->customdata);
  op->customdata = NULL;
}

void NODE_OT_backimage_move(wmOperatorType *ot)
{
  /* identifiers */
  ot->name = "Background Image Move";
  ot->description = "Move node backdrop";
  ot->idname = "NODE_OT_backimage_move";

  /* api callbacks */
  ot->invoke = snode_bg_viewmove_invoke;
  ot->modal = snode_bg_viewmove_modal;
  ot->poll = composite_node_active;
  ot->cancel = snode_bg_viewmove_cancel;

  /* flags */
  ot->flag = OPTYPE_BLOCKING | OPTYPE_GRAB_CURSOR_XY;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Background Image Zoom
 * \{ */

static int backimage_zoom_exec(bContext *C, wmOperator *op)
{
  SpaceNode *snode = CTX_wm_space_node(C);
  ARegion *region = CTX_wm_region(C);
  float fac = RNA_float_get(op->ptr, "factor");

  snode->zoom *= fac;
  ED_region_tag_redraw(region);
  WM_main_add_notifier(NC_NODE | ND_DISPLAY, NULL);
  WM_main_add_notifier(NC_SPACE | ND_SPACE_NODE_VIEW, NULL);

  return OPERATOR_FINISHED;
}

void NODE_OT_backimage_zoom(wmOperatorType *ot)
{

  /* identifiers */
  ot->name = "Background Image Zoom";
  ot->idname = "NODE_OT_backimage_zoom";
  ot->description = "Zoom in/out the background image";

  /* api callbacks */
  ot->exec = backimage_zoom_exec;
  ot->poll = composite_node_active;

  /* flags */
  ot->flag = OPTYPE_BLOCKING;

  /* internal */
  RNA_def_float(ot->srna, "factor", 1.2f, 0.0f, 10.0f, "Factor", "", 0.0f, 10.0f);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Background Image Fit
 * \{ */

static int backimage_fit_exec(bContext *C, wmOperator *UNUSED(op))
{
  Main *bmain = CTX_data_main(C);
  SpaceNode *snode = CTX_wm_space_node(C);
  ARegion *region = CTX_wm_region(C);

  Image *ima;
  ImBuf *ibuf;

  const float pad = 32.0f;

  void *lock;

  float facx, facy;

  ima = BKE_image_ensure_viewer(bmain, IMA_TYPE_COMPOSITE, "Viewer Node");
  ibuf = BKE_image_acquire_ibuf(ima, NULL, &lock);

  if ((ibuf == NULL) || (ibuf->x == 0) || (ibuf->y == 0)) {
    BKE_image_release_ibuf(ima, ibuf, lock);
    return OPERATOR_CANCELLED;
  }

  facx = 1.0f * (region->sizex - pad) / (ibuf->x * snode->zoom);
  facy = 1.0f * (region->sizey - pad) / (ibuf->y * snode->zoom);

  BKE_image_release_ibuf(ima, ibuf, lock);

  snode->zoom *= min_ff(facx, facy) * U.dpi_fac;

  snode->xof = 0;
  snode->yof = 0;

  ED_region_tag_redraw(region);
  WM_main_add_notifier(NC_NODE | ND_DISPLAY, NULL);
  WM_main_add_notifier(NC_SPACE | ND_SPACE_NODE_VIEW, NULL);

  return OPERATOR_FINISHED;
}

void NODE_OT_backimage_fit(wmOperatorType *ot)
{

  /* identifiers */
  ot->name = "Background Image Fit";
  ot->idname = "NODE_OT_backimage_fit";
  ot->description = "Fit the background image to the view";

  /* api callbacks */
  ot->exec = backimage_fit_exec;
  ot->poll = composite_node_active;

  /* flags */
  ot->flag = OPTYPE_BLOCKING;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Sample Backdrop Operator
 * \{ */

typedef struct ImageSampleInfo {
  ARegionType *art;
  void *draw_handle;
  int x, y;
  int channels;

  uchar col[4];
  float colf[4];
  float linearcol[4];

  int z;
  float zf;

  int *zp;
  float *zfp;

  int draw;
  int color_manage;
} ImageSampleInfo;

static void sample_draw(const bContext *C, ARegion *region, void *arg_info)
{
  Scene *scene = CTX_data_scene(C);
  ImageSampleInfo *info = arg_info;

  if (info->draw) {
    ED_image_draw_info(scene,
                       region,
                       info->color_manage,
                       false,
                       info->channels,
                       info->x,
                       info->y,
                       info->col,
                       info->colf,
                       info->linearcol,
                       info->zp,
                       info->zfp);
  }
}

/* Returns mouse position in image space. */
bool ED_space_node_get_position(
    Main *bmain, SpaceNode *snode, struct ARegion *ar, const int mval[2], float fpos[2])
{
  if (!ED_node_is_compositor(snode) || (snode->flag & SNODE_BACKDRAW) == 0) {
    return false;
  }

  void *lock;
  Image *ima = BKE_image_ensure_viewer(bmain, IMA_TYPE_COMPOSITE, "Viewer Node");
  ImBuf *ibuf = BKE_image_acquire_ibuf(ima, NULL, &lock);
  if (!ibuf) {
    BKE_image_release_ibuf(ima, ibuf, lock);
    return false;
  }

  /* map the mouse coords to the backdrop image space */
  float bufx = ibuf->x * snode->zoom;
  float bufy = ibuf->y * snode->zoom;
  fpos[0] = (bufx > 0.0f ? ((float)mval[0] - 0.5f * ar->winx - snode->xof) / bufx + 0.5f : 0.0f);
  fpos[1] = (bufy > 0.0f ? ((float)mval[1] - 0.5f * ar->winy - snode->yof) / bufy + 0.5f : 0.0f);

  BKE_image_release_ibuf(ima, ibuf, lock);
  return true;
}

/* Returns color in linear space, matching ED_space_image_color_sample().
 * And here we've got recursion in the comments tips...
 */
bool ED_space_node_color_sample(
    Main *bmain, SpaceNode *snode, ARegion *region, const int mval[2], float r_col[3])
{
  void *lock;
  Image *ima;
  ImBuf *ibuf;
  float fx, fy, bufx, bufy;
  bool ret = false;

  if (!ED_node_is_compositor(snode) || (snode->flag & SNODE_BACKDRAW) == 0) {
    /* use viewer image for color sampling only if we're in compositor tree
     * with backdrop enabled
     */
    return false;
  }

  ima = BKE_image_ensure_viewer(bmain, IMA_TYPE_COMPOSITE, "Viewer Node");
  ibuf = BKE_image_acquire_ibuf(ima, NULL, &lock);
  if (!ibuf) {
    return false;
  }

  /* map the mouse coords to the backdrop image space */
  bufx = ibuf->x * snode->zoom;
  bufy = ibuf->y * snode->zoom;
  fx = (bufx > 0.0f ? ((float)mval[0] - 0.5f * region->winx - snode->xof) / bufx + 0.5f : 0.0f);
  fy = (bufy > 0.0f ? ((float)mval[1] - 0.5f * region->winy - snode->yof) / bufy + 0.5f : 0.0f);

  if (fx >= 0.0f && fy >= 0.0f && fx < 1.0f && fy < 1.0f) {
    const float *fp;
    uchar *cp;
    int x = (int)(fx * ibuf->x), y = (int)(fy * ibuf->y);

    CLAMP(x, 0, ibuf->x - 1);
    CLAMP(y, 0, ibuf->y - 1);

    if (ibuf->rect_float) {
      fp = (ibuf->rect_float + (ibuf->channels) * (y * ibuf->x + x));
      /* #IB_PROFILE_NONE is default but in fact its linear. */
      copy_v3_v3(r_col, fp);
      ret = true;
    }
    else if (ibuf->rect) {
      cp = (uchar *)(ibuf->rect + y * ibuf->x + x);
      rgb_uchar_to_float(r_col, cp);
      IMB_colormanagement_colorspace_to_scene_linear_v3(r_col, ibuf->rect_colorspace);
      ret = true;
    }
  }

  BKE_image_release_ibuf(ima, ibuf, lock);

  return ret;
}

static void sample_apply(bContext *C, wmOperator *op, const wmEvent *event)
{
  Main *bmain = CTX_data_main(C);
  SpaceNode *snode = CTX_wm_space_node(C);
  ARegion *region = CTX_wm_region(C);
  ImageSampleInfo *info = op->customdata;
  void *lock;
  Image *ima;
  ImBuf *ibuf;
  float fx, fy, bufx, bufy;

  ima = BKE_image_ensure_viewer(bmain, IMA_TYPE_COMPOSITE, "Viewer Node");
  ibuf = BKE_image_acquire_ibuf(ima, NULL, &lock);
  if (!ibuf) {
    info->draw = 0;
    return;
  }

  if (!ibuf->rect) {
    IMB_rect_from_float(ibuf);
  }

  /* map the mouse coords to the backdrop image space */
  bufx = ibuf->x * snode->zoom;
  bufy = ibuf->y * snode->zoom;
  fx = (bufx > 0.0f ? ((float)event->mval[0] - 0.5f * region->winx - snode->xof) / bufx + 0.5f :
                      0.0f);
  fy = (bufy > 0.0f ? ((float)event->mval[1] - 0.5f * region->winy - snode->yof) / bufy + 0.5f :
                      0.0f);

  if (fx >= 0.0f && fy >= 0.0f && fx < 1.0f && fy < 1.0f) {
    const float *fp;
    uchar *cp;
    int x = (int)(fx * ibuf->x), y = (int)(fy * ibuf->y);

    CLAMP(x, 0, ibuf->x - 1);
    CLAMP(y, 0, ibuf->y - 1);

    info->x = x;
    info->y = y;
    info->draw = 1;
    info->channels = ibuf->channels;

    info->zp = NULL;
    info->zfp = NULL;

    if (ibuf->rect) {
      cp = (uchar *)(ibuf->rect + y * ibuf->x + x);

      info->col[0] = cp[0];
      info->col[1] = cp[1];
      info->col[2] = cp[2];
      info->col[3] = cp[3];

      info->colf[0] = (float)cp[0] / 255.0f;
      info->colf[1] = (float)cp[1] / 255.0f;
      info->colf[2] = (float)cp[2] / 255.0f;
      info->colf[3] = (float)cp[3] / 255.0f;

      copy_v4_v4(info->linearcol, info->colf);
      IMB_colormanagement_colorspace_to_scene_linear_v4(
          info->linearcol, false, ibuf->rect_colorspace);

      info->color_manage = true;
    }
    if (ibuf->rect_float) {
      fp = (ibuf->rect_float + (ibuf->channels) * (y * ibuf->x + x));

      info->colf[0] = fp[0];
      info->colf[1] = fp[1];
      info->colf[2] = fp[2];
      info->colf[3] = fp[3];

      info->color_manage = true;
    }

    if (ibuf->zbuf) {
      info->z = ibuf->zbuf[y * ibuf->x + x];
      info->zp = &info->z;
    }
    if (ibuf->zbuf_float) {
      info->zf = ibuf->zbuf_float[y * ibuf->x + x];
      info->zfp = &info->zf;
    }

    ED_node_sample_set(info->colf);
  }
  else {
    info->draw = 0;
    ED_node_sample_set(NULL);
  }

  BKE_image_release_ibuf(ima, ibuf, lock);

  ED_area_tag_redraw(CTX_wm_area(C));
}

static void sample_exit(bContext *C, wmOperator *op)
{
  ImageSampleInfo *info = op->customdata;

  ED_node_sample_set(NULL);
  ED_region_draw_cb_exit(info->art, info->draw_handle);
  ED_area_tag_redraw(CTX_wm_area(C));
  MEM_freeN(info);
}

static int sample_invoke(bContext *C, wmOperator *op, const wmEvent *event)
{
  SpaceNode *snode = CTX_wm_space_node(C);
  ARegion *region = CTX_wm_region(C);
  ImageSampleInfo *info;

  if (!ED_node_is_compositor(snode) || !(snode->flag & SNODE_BACKDRAW)) {
    return OPERATOR_CANCELLED;
  }

  info = MEM_callocN(sizeof(ImageSampleInfo), "ImageSampleInfo");
  info->art = region->type;
  info->draw_handle = ED_region_draw_cb_activate(
      region->type, sample_draw, info, REGION_DRAW_POST_PIXEL);
  op->customdata = info;

  sample_apply(C, op, event);

  WM_event_add_modal_handler(C, op);

  return OPERATOR_RUNNING_MODAL;
}

static int sample_modal(bContext *C, wmOperator *op, const wmEvent *event)
{
  switch (event->type) {
    case LEFTMOUSE:
    case RIGHTMOUSE: /* XXX hardcoded */
      if (event->val == KM_RELEASE) {
        sample_exit(C, op);
        return OPERATOR_CANCELLED;
      }
      break;
    case MOUSEMOVE:
      sample_apply(C, op, event);
      break;
  }

  return OPERATOR_RUNNING_MODAL;
}

static void sample_cancel(bContext *C, wmOperator *op)
{
  sample_exit(C, op);
}

void NODE_OT_backimage_sample(wmOperatorType *ot)
{
  /* identifiers */
  ot->name = "Backimage Sample";
  ot->idname = "NODE_OT_backimage_sample";
  ot->description = "Use mouse to sample background image";

  /* api callbacks */
  ot->invoke = sample_invoke;
  ot->modal = sample_modal;
  ot->cancel = sample_cancel;
  ot->poll = ED_operator_node_active;

  /* flags */
  ot->flag = OPTYPE_BLOCKING;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Edge Pan Operator Utilties
 * \{ */

/** Distance from the edge of the region within which to start panning. */
#define EDGE_PAN_REGION_PAD (U.widget_unit)
/** Speed factor in pixels per second per pixel of distance from edge pan zone beginning. */
#define EDGE_PAN_SPEED_PER_PIXEL (10.0f * (float)U.dpi_fac)
/** Delay before drag panning in seconds. */
#define EDGE_PAN_DELAY 1.0f

bool node_edge_pan_poll(bContext *C)
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

void node_edge_pan_init(bContext *C, NodeViewPanData *vpd)
{
  if (!node_edge_pan_poll(C)) {
    return;
  }

  /* set pointers to owners */
  vpd->screen = CTX_wm_screen(C);
  vpd->area = CTX_wm_area(C);
  vpd->region = CTX_wm_region(C);
  vpd->v2d = &vpd->region->v2d;

  /* calculate translation factor - based on size of view */
  const float winx = (float)(BLI_rcti_size_x(&vpd->region->winrct) + 1);
  const float winy = (float)(BLI_rcti_size_y(&vpd->region->winrct) + 1);
  vpd->facx = (BLI_rctf_size_x(&vpd->v2d->cur)) / winx;
  vpd->facy = (BLI_rctf_size_y(&vpd->v2d->cur)) / winy;
}

void node_edge_pan_start(bContext *UNUSED(C), NodeViewPanData *vpd, wmOperator *UNUSED(op))
{
  vpd->edge_pan_start_time_x = 0.0;
  vpd->edge_pan_start_time_y = 0.0;
  vpd->edge_pan_last_time = PIL_check_seconds_timer();
}

void node_edge_pan_apply_ex(bContext *C, NodeViewPanData *vpd, float dx, float dy)
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

/**
 * Reset the edge pan timers if the mouse isn't in the scroll zone and
 * start the timers when the mouse enters a scroll zone.
 */
static void edge_pan_manage_delay_timers(NodeViewPanData *vpd,
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

static float edge_pan_speed(NodeViewPanData *vpd,
                            int event_loc,
                            bool x_dir,
                            const double current_time)
{
  ARegion *region = vpd->region;

  /* Find the distance from the start of the drag zone. */
  const int min = (x_dir ? region->winrct.xmin : region->winrct.ymin) + EDGE_PAN_REGION_PAD;
  const int max = (x_dir ? region->winrct.xmax : region->winrct.ymax) - EDGE_PAN_REGION_PAD;
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
  const float delay_factor = smootherstep(EDGE_PAN_DELAY, (float)(current_time - start_time));

  return distance * EDGE_PAN_SPEED_PER_PIXEL * delay_factor;
}

void node_edge_pan_properties(wmOperatorType* ot)
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

void node_edge_pan_apply_op(bContext *C,
                            NodeViewPanData *vpd,
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
    /* Find whether the mouse is beyond X and Y edges. */
    if (event->x > region->winrct.xmax - EDGE_PAN_REGION_PAD) {
      pan_dir_x = 1;
    }
    else if (event->x < region->winrct.xmin + EDGE_PAN_REGION_PAD) {
      pan_dir_x = -1;
    }
    if (event->y > region->winrct.ymax - EDGE_PAN_REGION_PAD) {
      pan_dir_y = 1;
    }
    else if (event->y < region->winrct.ymin + EDGE_PAN_REGION_PAD) {
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
  node_edge_pan_apply_ex(C, vpd, dx, dy);
}

#undef EDGE_PAN_REGION_PAD
#undef EDGE_PAN_SPEED_PER_PIXEL
#undef EDGE_PAN_DELAY

/** \} */

