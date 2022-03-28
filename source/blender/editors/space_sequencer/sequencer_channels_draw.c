/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

/** \file
 * \ingroup sequencer
 */

#include "DNA_scene_types.h"
#include "DNA_screen_types.h"

#include "BKE_context.h"

#include "BLI_blenlib.h"
#include "BLI_utildefines.h"

#include "ED_screen.h"

#include "GPU_framebuffer.h"
#include "GPU_immediate.h"
#include "GPU_immediate_util.h"
#include "GPU_matrix.h"
#include "GPU_state.h"
#include "GPU_vertex_buffer.h"
#include "GPU_viewport.h"

#include "RNA_access.h"
#include "RNA_prototypes.h"

#include "SEQ_channels.h"
#include "SEQ_sequencer.h"
#include "SEQ_time.h"

#include "UI_interface.h"
#include "UI_resources.h"
#include "UI_view2d.h"

#include "WM_api.h"

/* Own include. */
#include "sequencer_intern.h"

static ARegion *timeline_region_get(ScrArea *area)
{
  LISTBASE_FOREACH (ARegion *, region, &area->regionbase) {
    if (region->regiontype == RGN_TYPE_WINDOW) {
      return region;
    }
  }

  BLI_assert_unreachable();
  return NULL;
}

static float draw_offset_get(View2D *timeline_region_v2d)
{
  return timeline_region_v2d->cur.ymin;
}

static float channel_height_pixelspace_get(View2D *timeline_region_v2d)
{
  return UI_view2d_view_to_region_y(timeline_region_v2d, 1.0f) -
         UI_view2d_view_to_region_y(timeline_region_v2d, 0.0f);
}

static float frame_width_pixelspace_get(View2D *timeline_region_v2d)
{

  return UI_view2d_view_to_region_x(timeline_region_v2d, 1.0f) -
         UI_view2d_view_to_region_x(timeline_region_v2d, 0.0f);
}

static float icon_width_get(SeqChannelDrawContext *context)
{
  return (U.widget_unit * 0.8 * context->scale);
}

static float widget_y_offset(SeqChannelDrawContext *context)
{
  return (((context->channel_height / context->scale) - icon_width_get(context))) / 2;
}

static float channel_index_y_min(SeqChannelDrawContext *context, const int index)
{
  float y = (index - context->draw_offset) * context->channel_height;
  y /= context->scale;
  return y;
}

static void displayed_channel_range_get(SeqChannelDrawContext *context, int channel_range[2])
{
  /* Channel 0 is not usable, so should never be drawn. */
  channel_range[0] = max_ii(1, floor(context->timeline_region_v2d->cur.ymin));
  channel_range[1] = ceil(context->timeline_region_v2d->cur.ymax);

  rctf strip_boundbox;
  BLI_rctf_init(&strip_boundbox, 0.0f, 0.0f, 1.0f, 7);
  SEQ_timeline_expand_boundbox(context->seqbase, &strip_boundbox);
  CLAMP(channel_range[0], strip_boundbox.ymin, strip_boundbox.ymax);
  CLAMP(channel_range[1], strip_boundbox.ymin, strip_boundbox.ymax);
}

static float draw_channel_widget_hide(SeqChannelDrawContext *context,
                                      uiBlock *block,
                                      int channel_index,
                                      float offset)
{
  float y = channel_index_y_min(context, channel_index) + widget_y_offset(context);

  const float width = icon_width_get(context);
  SeqTimelineChannel *channel = SEQ_channel_get_by_index(context->channels, channel_index);
  const int icon = SEQ_channel_is_muted(channel) ? ICON_CHECKBOX_DEHLT : ICON_CHECKBOX_HLT;
  const char *tooltip = BLI_sprintfN(
      "%s channel %d", SEQ_channel_is_muted(channel) ? "Unmute" : "Mute", channel_index);

  PointerRNA ptr;
  RNA_pointer_create(&context->scene->id, &RNA_SequenceTimelineChannel, channel, &ptr);
  PropertyRNA *hide_prop = RNA_struct_type_find_property(&RNA_SequenceTimelineChannel, "mute");

  UI_block_emboss_set(block, UI_EMBOSS_NONE);
  uiDefIconButR_prop(block,
                     UI_BTYPE_TOGGLE,
                     1,
                     icon,
                     context->v2d->cur.xmax / context->scale - offset,
                     y,
                     width,
                     width,
                     &ptr,
                     hide_prop,
                     0,
                     0,
                     0,
                     0,
                     0,
                     tooltip);

  return width;
}

static float draw_channel_widget_lock(SeqChannelDrawContext *context,
                                      uiBlock *block,
                                      int channel_index,
                                      float offset)
{

  float y = channel_index_y_min(context, channel_index) + widget_y_offset(context);
  const float width = icon_width_get(context);

  SeqTimelineChannel *channel = SEQ_channel_get_by_index(context->channels, channel_index);
  const int icon = SEQ_channel_is_locked(channel) ? ICON_LOCKED : ICON_UNLOCKED;
  const char *tooltip = BLI_sprintfN(
      "%s channel %d", SEQ_channel_is_locked(channel) ? "Unlock" : "Lock", channel_index);

  PointerRNA ptr;
  RNA_pointer_create(&context->scene->id, &RNA_SequenceTimelineChannel, channel, &ptr);
  PropertyRNA *hide_prop = RNA_struct_type_find_property(&RNA_SequenceTimelineChannel, "lock");

  UI_block_emboss_set(block, UI_EMBOSS_NONE);
  uiDefIconButR_prop(block,
                     UI_BTYPE_TOGGLE,
                     1,
                     icon,
                     context->v2d->cur.xmax / context->scale - offset,
                     y,
                     width,
                     width,
                     &ptr,
                     hide_prop,
                     0,
                     0,
                     0,
                     0,
                     0,
                     tooltip);

  return width;
}

static bool channel_is_being_renamed(SpaceSeq *sseq, int channel_index)
{
  return sseq->runtime.rename_channel_index == channel_index;
}

static float text_size_get(SeqChannelDrawContext *context)
{
  const uiStyle *style = UI_style_get_dpi();
  return UI_fontstyle_height_max(&style->widget) * 1.5f * context->scale;
}

/* Todo: decide what gets priority - label or buttons */
static void label_rect_init(SeqChannelDrawContext *context,
                            int channel_index,
                            float used_width,
                            rctf *r_rect)
{
  float text_size = text_size_get(context);
  float margin = (context->channel_height / context->scale - text_size) / 2.0f;
  float y = channel_index_y_min(context, channel_index) + margin;

  float margin_x = icon_width_get(context) * 0.65;
  float width = max_ff(0.0f, context->v2d->cur.xmax / context->scale - used_width);

  /* Text input has own margin. Prevent text jumping around and use as much space as possible. */
  if (channel_is_being_renamed(CTX_wm_space_seq(context->C), channel_index)) {
    float input_box_margin = icon_width_get(context) * 0.5f;
    margin_x -= input_box_margin;
    width += input_box_margin;
  }

  BLI_rctf_init(r_rect, margin_x, margin_x + width, y, y + text_size);
}

static void draw_channel_labels(SeqChannelDrawContext *context,
                                uiBlock *block,
                                int channel_index,
                                float used_width)
{
  SpaceSeq *sseq = CTX_wm_space_seq(context->C);
  rctf rect;
  label_rect_init(context, channel_index, used_width, &rect);

  if (BLI_rctf_size_y(&rect) <= 1.0f || BLI_rctf_size_x(&rect) <= 1.0f) {
    return;
  }

  if (channel_is_being_renamed(sseq, channel_index)) {
    SeqTimelineChannel *channel = SEQ_channel_get_by_index(context->channels, channel_index);
    PointerRNA ptr = {NULL};
    RNA_pointer_create(&context->scene->id, &RNA_SequenceTimelineChannel, channel, &ptr);
    PropertyRNA *prop = RNA_struct_name_property(ptr.type);

    UI_block_emboss_set(block, UI_EMBOSS);
    uiBut *but = uiDefButR(block,
                           UI_BTYPE_TEXT,
                           1,
                           "",
                           rect.xmin,
                           rect.ymin,
                           BLI_rctf_size_x(&rect),
                           BLI_rctf_size_y(&rect),
                           &ptr,
                           RNA_property_identifier(prop),
                           -1,
                           0,
                           0,
                           0,
                           0,
                           NULL);
    UI_block_emboss_set(block, UI_EMBOSS_NONE);

    if (UI_but_active_only(context->C, context->region, block, but) == false) {
      sseq->runtime.rename_channel_index = 0;
    }

    WM_event_add_notifier(context->C, NC_SCENE | ND_SEQUENCER, context->scene);
  }
  else {
    const char *label = SEQ_channel_name_get(context->channels, channel_index);
    uiDefBut(block,
             UI_BTYPE_LABEL,
             0,
             label,
             rect.xmin,
             rect.ymin,
             rect.xmax - rect.xmin,
             (rect.ymax - rect.ymin),
             NULL,
             0,
             0,
             0,
             0,
             NULL);
  }
}

/* Todo: different text/buttons alignment */
static void draw_channel_header(SeqChannelDrawContext *context, uiBlock *block, int channel_index)
{
  float offset = icon_width_get(context) * 1.5f;
  offset += draw_channel_widget_lock(context, block, channel_index, offset);
  offset += draw_channel_widget_hide(context, block, channel_index, offset);

  draw_channel_labels(context, block, channel_index, offset);
}

static void draw_channel_headers(SeqChannelDrawContext *context)
{
  GPU_matrix_push();
  wmOrtho2_pixelspace(context->region->winx / context->scale,
                      context->region->winy / context->scale);
  uiBlock *block = UI_block_begin(context->C, context->region, __func__, UI_EMBOSS);

  int channel_range[2];
  displayed_channel_range_get(context, channel_range);

  for (int channel = channel_range[0]; channel <= channel_range[1]; channel++) {
    draw_channel_header(context, block, channel);
  }

  UI_block_end(context->C, block);
  UI_block_draw(context->C, block);

  GPU_matrix_pop();
}

static void draw_background(SeqChannelDrawContext *context)
{
  UI_ThemeClearColor(TH_BACK);
}

void channel_draw_context_init(const bContext *C,
                               ARegion *region,
                               SeqChannelDrawContext *r_context)
{
  r_context->C = C;
  r_context->area = CTX_wm_area(C);
  r_context->region = region;
  r_context->v2d = &region->v2d;
  r_context->scene = CTX_data_scene(C);
  r_context->ed = SEQ_editing_get(r_context->scene);
  r_context->seqbase = SEQ_active_seqbase_get(r_context->ed);
  r_context->channels = SEQ_channels_active_get(r_context->ed);
  r_context->timeline_region = timeline_region_get(CTX_wm_area(C));
  r_context->timeline_region_v2d = &r_context->timeline_region->v2d;

  r_context->channel_height = channel_height_pixelspace_get(r_context->timeline_region_v2d);
  r_context->frame_width = frame_width_pixelspace_get(r_context->timeline_region_v2d);
  r_context->draw_offset = draw_offset_get(r_context->timeline_region_v2d);

  r_context->scale = min_ff(r_context->channel_height / (U.widget_unit * 0.6), 1);
}

void draw_channels(const bContext *C, ARegion *region)
{
  SeqChannelDrawContext context;
  channel_draw_context_init(C, region, &context);

  UI_view2d_view_ortho(context.v2d);

  draw_background(&context);
  draw_channel_headers(&context);

  UI_view2d_view_restore(C);
}
