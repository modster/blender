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
 * Copyright 2021, Blender Foundation.
 */

#include "BLI_hash_tables.hh"
#include "BLI_rect.h"
#include "BLI_vector.hh"

#include "RE_pipeline.h"

#include "eevee_instance.hh"
#include "eevee_renderpasses.hh"

namespace blender::eevee {

class Instance;

/* -------------------------------------------------------------------- */
/** \name RenderPasses
 * \{ */

void RenderPasses::init(const int extent[2], const rcti *output_rect)
{
  const Scene *scene = inst_.scene;

  eRenderPassBit enabled_passes;
  if (inst_.render_layer) {
    enabled_passes = to_render_passes_bits(inst_.render_layer->passflag);
    /* Cannot output motion vectors when using motion blur. */
    if (scene->eevee.flag & SCE_EEVEE_MOTION_BLUR_ENABLED) {
      enabled_passes &= ~RENDERPASS_VECTOR;
    }
  }
  else {
    BLI_assert(inst_.v3d);
    enabled_passes = to_render_passes_bits(inst_.v3d->shading.render_pass);
    /* We need the depth pass for compositing overlays or GPencil. */
    if (!DRW_state_is_scene_render()) {
      enabled_passes |= RENDERPASS_DEPTH;
    }
  }

  const bool use_log_encoding = scene->eevee.flag & SCE_EEVEE_FILM_LOG_ENCODING;

  rcti fallback_rect;
  if (BLI_rcti_is_empty(output_rect)) {
    BLI_rcti_init(&fallback_rect, 0, extent[0], 0, extent[1]);
    output_rect = &fallback_rect;
  }

  /* HACK to iterate over all passes. */
  enabled_passes_ = static_cast<eRenderPassBit>(0xFFFFFFFF);
  for (RenderPassItem rpi : *this) {
    bool enable = (enabled_passes & rpi.pass_bit) != 0;
    if (enable && rpi.film == nullptr) {
      rpi.film = new Film(inst_,
                          to_render_passes_data_type(rpi.pass_bit, use_log_encoding),
                          to_render_passes_name(rpi.pass_bit));
    }
    else if (!enable && rpi.film != nullptr) {
      /* Delete unused passes. */
      delete rpi.film;
      rpi.film = nullptr;
    }

    if (rpi.film) {
      rpi.film->init(extent, output_rect);
    }
  }

  enabled_passes_ = enabled_passes;
}

/** \} */

}  // namespace blender::eevee
