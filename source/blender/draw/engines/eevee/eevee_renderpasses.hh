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

#pragma once

#include "BLI_vector.hh"

#include "RE_pipeline.h"

#include "eevee_film.hh"

namespace blender::eevee {

/* -------------------------------------------------------------------- */
/** \name eRenderPassBit
 *
 * This enum might seems redundant but there is an opportunity to use it for internal debug passes.
 * \{ */

enum eRenderPassBit {
  RENDERPASS_NONE = 0,
  RENDERPASS_COMBINED = (1 << 0),
  RENDERPASS_DEPTH = (1 << 1),
  RENDERPASS_NORMAL = (1 << 2),
  /** Used for iterator. */
  RENDERPASS_MAX,
};

ENUM_OPERATORS(eRenderPassBit, RENDERPASS_NORMAL)

static eRenderPassBit to_render_passes_bits(int i_rpasses)
{
  eRenderPassBit rpasses = RENDERPASS_NONE;
  SET_FLAG_FROM_TEST(rpasses, i_rpasses & SCE_PASS_COMBINED, RENDERPASS_COMBINED);
  SET_FLAG_FROM_TEST(rpasses, i_rpasses & SCE_PASS_Z, RENDERPASS_DEPTH);
  SET_FLAG_FROM_TEST(rpasses, i_rpasses & SCE_PASS_NORMAL, RENDERPASS_NORMAL);
  return rpasses;
}

static const char *to_render_passes_name(eRenderPassBit rpass)
{
  switch (rpass) {
    case RENDERPASS_COMBINED:
      return RE_PASSNAME_COMBINED;
    case RENDERPASS_DEPTH:
      return RE_PASSNAME_Z;
    case RENDERPASS_NORMAL:
      return RE_PASSNAME_NORMAL;
    default:
      BLI_assert(0);
      return "";
  }
}

static eFilmDataType to_render_passes_data_type(eRenderPassBit rpass, const bool use_log_encoding)
{
  switch (rpass) {
    case RENDERPASS_COMBINED:
      return (use_log_encoding) ? FILM_DATA_COLOR_LOG : FILM_DATA_COLOR;
    case RENDERPASS_DEPTH:
      return FILM_DATA_DEPTH;
    case RENDERPASS_NORMAL:
      return FILM_DATA_NORMAL;
    default:
      BLI_assert(0);
      return FILM_DATA_COLOR;
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name RenderPasses
 * \{ */

typedef struct RenderPasses {
 public:
  /** Film for each render pass. A nullptr means the pass is not needed. */
  Film *combined = nullptr;
  Film *depth = nullptr;
  Film *normal = nullptr;
  Vector<Film *> aovs;

 private:
  ShaderModule &shaders_;
  Camera &camera_;
  Sampling &sampling_;
  eRenderPassBit enabled_passes_ = RENDERPASS_NONE;

 public:
  RenderPasses(ShaderModule &shaders, Camera &camera, Sampling &sampling)
      : shaders_(shaders), camera_(camera), sampling_(sampling){};

  ~RenderPasses()
  {
    delete combined;
    delete depth;
    delete normal;
  }

  void init(const Scene *scene,
            const RenderLayer *render_layer,
            const View3D *v3d,
            const int extent[2],
            const rcti *output_rect)
  {
    if (render_layer) {
      enabled_passes_ = to_render_passes_bits(render_layer->passflag);
    }
    else {
      BLI_assert(v3d);
      enabled_passes_ = to_render_passes_bits(v3d->shading.render_pass);
      /* We need the depth pass for compositing overlays or GPencil. */
      if (!DRW_state_is_scene_render()) {
        enabled_passes_ |= RENDERPASS_DEPTH;
      }
    }

    const bool use_log_encoding = scene->eevee.flag & SCE_EEVEE_FILM_LOG_ENCODING;

    rcti fallback_rect;
    if (BLI_rcti_is_empty(output_rect)) {
      BLI_rcti_init(&fallback_rect, 0, extent[0], 0, extent[1]);
      output_rect = &fallback_rect;
    }

    for (int64_t i = 1; i < RENDERPASS_MAX; i <<= 1) {
      eRenderPassBit render_pass = static_cast<eRenderPassBit>(i);
      Film *&film = this->render_pass_bit_to_film_p(render_pass);

      bool enable = (enabled_passes_ & render_pass) != 0;
      if (enable && film == nullptr) {
        film = new Film(shaders_,
                        camera_,
                        sampling_,
                        to_render_passes_data_type(render_pass, use_log_encoding),
                        to_render_passes_name(render_pass));
      }
      else if (!enable && film != nullptr) {
        /* Delete unused passes. */
        delete film;
        film = nullptr;
      }

      if (film) {
        film->init(extent, output_rect);
      }
    }
  }

  void sync(void)
  {
    for (int64_t i = 1; i < RENDERPASS_MAX; i <<= 1) {
      eRenderPassBit render_pass = static_cast<eRenderPassBit>(i);
      Film *film = this->render_pass_bit_to_film_p(render_pass);

      if (film) {
        film->sync();
      }
    }
  }

  void resolve_viewport(DefaultFramebufferList *dfbl)
  {
    for (int64_t i = 1; i < RENDERPASS_MAX; i <<= 1) {
      eRenderPassBit render_pass = static_cast<eRenderPassBit>(i);
      Film *film = this->render_pass_bit_to_film_p(render_pass);

      if (film) {
        if (render_pass == RENDERPASS_DEPTH) {
          film->resolve_viewport(dfbl->depth_only_fb);
        }
        else {
          /* Ensures only one color render pass is enabled. */
          BLI_assert((enabled_passes_ & ~RENDERPASS_DEPTH) == render_pass);
          film->resolve_viewport(dfbl->color_only_fb);
        }
      }
    }
  }

  void read_result(RenderLayer *render_layer, const char *view_name)
  {
    for (int64_t i = 1; i < RENDERPASS_MAX; i <<= 1) {
      eRenderPassBit render_pass = static_cast<eRenderPassBit>(i);
      Film *film = this->render_pass_bit_to_film_p(render_pass);

      if (film) {
        const char *pass_name = to_render_passes_name(render_pass);
        RenderPass *rp = RE_pass_find_by_name(render_layer, pass_name, view_name);
        if (rp) {
          film->read_result(rp->rect);
        }
      }
    }
  }

 private:
  Film *&render_pass_bit_to_film_p(eRenderPassBit rpass)
  {
    switch (rpass) {
      case RENDERPASS_COMBINED:
        return combined;
      case RENDERPASS_DEPTH:
        return depth;
      case RENDERPASS_NORMAL:
        return normal;
      default:
        BLI_assert(0);
        return combined;
    }
  }
} RenderPasses;

/** \} */

}  // namespace blender::eevee
