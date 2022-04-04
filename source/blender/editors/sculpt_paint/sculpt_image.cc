/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */
#include "DNA_object_types.h"

#include "BKE_context.h"
#include "BKE_paint.h"
#include "BKE_pbvh.h"
#include "BKE_screen.h"

#include "WM_message.h"
#include "WM_types.h"

extern "C" {

static void sculpt_pbvh_do_msg_tag_rebuild_pixels(struct bContext *C,
                                                  wmMsgSubscribeKey *UNUSED(msg_key),
                                                  wmMsgSubscribeValue *UNUSED(msg_val))
{
  Object *ob = CTX_data_active_object(C);
  if (ob == nullptr) {
    return;
  }
  SculptSession *ss = ob->sculpt;
  if (ss == nullptr) {
    return;
  }
  BKE_pbvh_mark_update_pixels(ss->pbvh);
}

void ED_sculpt_pbvh_message_subscribe(const struct wmRegionMessageSubscribeParams *params)
{
  struct wmMsgBus *mbus = params->message_bus;

  wmMsgSubscribeValue notify_msg = {
      .owner = NULL,
      .user_data = NULL,
      .notify = sculpt_pbvh_do_msg_tag_rebuild_pixels,
  };
  WM_msg_subscribe_rna_anon_prop(mbus, Object, active_material_index, &notify_msg);
  WM_msg_subscribe_rna_anon_prop(mbus, MeshUVLoopLayer, active, &notify_msg);
  WM_msg_subscribe_rna_anon_type(mbus, Image, &notify_msg);
  WM_msg_subscribe_rna_anon_type(mbus, UDIMTile, &notify_msg);
}
}