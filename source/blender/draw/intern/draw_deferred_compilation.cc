/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2016 Blender Foundation. */

/** \file
 * \ingroup draw
 */

#include "PIL_time.h"

#include "DRW_engine.h"

#include "draw_manager.h"

#define USE_DEFERRED_COMPILATION 1
/* In seconds. */
#define COMPILER_TIME_ALLOWED_PER_REDRAW (1.0 / 5.0)

/* -------------------------------------------------------------------- */
/** \name Deferred Compilation (DRW_deferred)
 *
 * Since compiling shader can take a long time, we do it in a non blocking
 * manner by .
 *
 * \{ */

void DRW_deferred_shader_add(GPUMaterial *mat, bool deferred)
{
  if (GPU_material_status(mat) != GPU_MAT_QUEUED) {
    return;
  }

  bool force_compile = false;
  /* Do not defer the compilation if we are rendering for image.
   * deferred rendering is only possible when `evil_C` is available */
  if (DST.draw_ctx.evil_C == NULL || DRW_state_is_image_render() || !USE_DEFERRED_COMPILATION ||
      !deferred) {
    force_compile = true;
  }

  bool compiler_time_exhauted = DST.compiler_time > COMPILER_TIME_ALLOWED_PER_REDRAW;
  if (compiler_time_exhauted && !force_compile) {
    /* Leave material in its own state. */
    printf("Bypass %f\n", DST.compiler_time);
    DRW_viewport_request_redraw();
    return;
  }

  double start_time = PIL_check_seconds_timer();

  GPU_material_compile(mat);

  double end_time = PIL_check_seconds_timer();

  DST.compiler_time += end_time - start_time;
}

/** \} */
