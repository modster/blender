/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

/** \file
 * \ingroup edsculpt
 */

#pragma once

extern "C" {
/* Forward declarations. */
struct bContext;
struct Object;
}

#include "BLI_math_vec_types.hh"

namespace blender::ed::sculpt_paint::image3d {
struct StrokeHandle;

struct StrokeHandle *stroke_new(bContext *C, Object *ob);
void stroke_update(struct StrokeHandle *stroke_handle, float2 prev_mouse, float2 mouse);
void stroke_free(struct StrokeHandle *stroke_handle);

}  // namespace blender::ed::sculpt_paint::image3d
