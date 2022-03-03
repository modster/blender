/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2021 Blender Foundation.
 */

#pragma once

#include "DRW_render.h"

#include "BLI_hash_tables.hh"
#include "BLI_vector.hh"

#include "RE_pipeline.h"

#include "eevee_film.hh"

namespace blender::eevee {

class Instance;

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
  RENDERPASS_VECTOR = (1 << 3),
  RENDERPASS_EMISSION = (1 << 4),
  RENDERPASS_DIFFUSE_COLOR = (1 << 5),
  RENDERPASS_SPECULAR_COLOR = (1 << 6),
  RENDERPASS_VOLUME_LIGHT = (1 << 7),
  /** Used for iterator. */
  RENDERPASS_MAX,
  RENDERPASS_ALL = ((RENDERPASS_MAX - 1) << 1) - 1,
};

ENUM_OPERATORS(eRenderPassBit, RENDERPASS_NORMAL)

static inline eRenderPassBit to_render_passes_bits(int i_rpasses)
{
  eRenderPassBit rpasses = RENDERPASS_NONE;
  SET_FLAG_FROM_TEST(rpasses, i_rpasses & SCE_PASS_COMBINED, RENDERPASS_COMBINED);
  SET_FLAG_FROM_TEST(rpasses, i_rpasses & SCE_PASS_Z, RENDERPASS_DEPTH);
  SET_FLAG_FROM_TEST(rpasses, i_rpasses & SCE_PASS_NORMAL, RENDERPASS_NORMAL);
  SET_FLAG_FROM_TEST(rpasses, i_rpasses & SCE_PASS_VECTOR, RENDERPASS_VECTOR);
  SET_FLAG_FROM_TEST(rpasses, i_rpasses & SCE_PASS_EMIT, RENDERPASS_EMISSION);
  SET_FLAG_FROM_TEST(rpasses, i_rpasses & SCE_PASS_DIFFUSE_COLOR, RENDERPASS_DIFFUSE_COLOR);
  SET_FLAG_FROM_TEST(rpasses, i_rpasses & SCE_PASS_GLOSSY_COLOR, RENDERPASS_SPECULAR_COLOR);
  /* RENDERPASS_VOLUME_LIGHT? */
  return rpasses;
}

static inline const char *to_render_passes_name(eRenderPassBit rpass)
{
  switch (rpass) {
    case RENDERPASS_COMBINED:
      return RE_PASSNAME_COMBINED;
    case RENDERPASS_DEPTH:
      return RE_PASSNAME_Z;
    case RENDERPASS_NORMAL:
      return RE_PASSNAME_NORMAL;
    case RENDERPASS_VECTOR:
      return RE_PASSNAME_VECTOR;
    case RENDERPASS_EMISSION:
      return RE_PASSNAME_EMIT;
    case RENDERPASS_DIFFUSE_COLOR:
      return RE_PASSNAME_DIFFUSE_COLOR;
    case RENDERPASS_SPECULAR_COLOR:
      return RE_PASSNAME_GLOSSY_COLOR;
    case RENDERPASS_VOLUME_LIGHT:
      return RE_PASSNAME_VOLUME_LIGHT;
    default:
      BLI_assert(0);
      return "";
  }
}

static inline eFilmDataType to_render_passes_data_type(eRenderPassBit rpass,
                                                       const bool use_log_encoding)
{
  switch (rpass) {
    case RENDERPASS_EMISSION:
    case RENDERPASS_DIFFUSE_COLOR:
    case RENDERPASS_SPECULAR_COLOR:
    case RENDERPASS_VOLUME_LIGHT:
    case RENDERPASS_COMBINED:
      return (use_log_encoding) ? FILM_DATA_COLOR_LOG : FILM_DATA_COLOR;
    case RENDERPASS_DEPTH:
      return FILM_DATA_DEPTH;
    case RENDERPASS_NORMAL:
      return FILM_DATA_NORMAL;
    case RENDERPASS_VECTOR:
      return FILM_DATA_MOTION;
    default:
      BLI_assert(0);
      return FILM_DATA_COLOR;
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name RenderPasses
 * \{ */

/**
 * Contains and manages each \c Film output for each render pass output.
 */
class RenderPasses {
 public:
  /** Film for each render pass. A nullptr means the pass is not needed. */
  Film *combined = nullptr;
  Film *depth = nullptr;
  Film *normal = nullptr;
  Film *vector = nullptr;
  Film *emission = nullptr;
  Film *diffuse_color = nullptr;
  Film *diffuse_light = nullptr;
  Film *specular_color = nullptr;
  Film *specular_light = nullptr;
  Film *volume_light = nullptr;
  Vector<Film *> aovs;
  /** View texture to render to. */
  TextureFromPool emission_tx = {"PassEmission"};
  TextureFromPool diffuse_color_tx = {"PassDiffuseColor"};
  TextureFromPool specular_color_tx = {"PassSpecularColor"};
  TextureFromPool volume_light_tx = {"PassVolumeLight"};

 private:
  Instance &inst_;

  eRenderPassBit enabled_passes_ = RENDERPASS_NONE;
  /* Maximum texture size. Since we use imageLoad/Store instead of framebuffer, we only need to
   * allocate the biggest texture. */
  int2 tmp_extent_ = int2(-1);

 public:
  RenderPasses(Instance &inst) : inst_(inst){};

  ~RenderPasses()
  {
    delete combined;
    delete depth;
    delete normal;
    delete vector;
    for (Film *&film : aovs) {
      delete film;
    }
  }

  void init(const int extent[2], const rcti *output_rect);

  void sync(void)
  {
    for (RenderPassItem rpi : *this) {
      rpi.film->sync();
    }
    emission_tx.sync();
    diffuse_color_tx.sync();
    specular_color_tx.sync();
    volume_light_tx.sync();
    tmp_extent_ = int2(-1);
  }

  void view_sync(int2 view_extent)
  {
    tmp_extent_ = math::max(tmp_extent_, view_extent);
  }

  void end_sync(void)
  {
    for (RenderPassItem rpi : *this) {
      rpi.film->end_sync();
    }
  }

  void acquire()
  {
    auto acquire_tmp_pass_buffer = [&](TextureFromPool &texture, bool enabled_pass) {
      texture.acquire(enabled_pass ? tmp_extent_ : int2(1), GPU_RGBA16F, (void *)&inst_);
    };
    acquire_tmp_pass_buffer(emission_tx, enabled_passes_ & RENDERPASS_EMISSION);
    acquire_tmp_pass_buffer(diffuse_color_tx, enabled_passes_ & RENDERPASS_DIFFUSE_COLOR);
    acquire_tmp_pass_buffer(specular_color_tx, enabled_passes_ & RENDERPASS_SPECULAR_COLOR);
    acquire_tmp_pass_buffer(volume_light_tx, enabled_passes_ & RENDERPASS_VOLUME_LIGHT);
  }

  void release()
  {
    emission_tx.release();
    diffuse_color_tx.release();
    specular_color_tx.release();
    volume_light_tx.release();
  }

  void resolve_viewport(DefaultFramebufferList *dfbl)
  {
    for (RenderPassItem rpi : *this) {
      if (rpi.pass_bit == RENDERPASS_DEPTH) {
        rpi.film->resolve_viewport(dfbl->depth_only_fb);
      }
      else {
        /* Ensures only one color render pass is enabled. */
        BLI_assert((enabled_passes_ & ~RENDERPASS_DEPTH) == rpi.pass_bit);
        rpi.film->resolve_viewport(dfbl->color_only_fb);
      }
    }
  }

  void read_result(RenderLayer *render_layer, const char *view_name)
  {
    for (RenderPassItem rpi : *this) {
      const char *pass_name = to_render_passes_name(rpi.pass_bit);
      RenderPass *rp = RE_pass_find_by_name(render_layer, pass_name, view_name);
      if (rp) {
        rpi.film->read_result(rp->rect);
      }
    }
  }

 private:
  constexpr Film *&render_pass_bit_to_film_p(eRenderPassBit rpass)
  {
    switch (rpass) {
      case RENDERPASS_COMBINED:
        return combined;
      case RENDERPASS_DEPTH:
        return depth;
      case RENDERPASS_NORMAL:
        return normal;
      case RENDERPASS_VECTOR:
        return vector;
      case RENDERPASS_EMISSION:
        return emission;
      case RENDERPASS_DIFFUSE_COLOR:
        return vector;
      case RENDERPASS_SPECULAR_COLOR:
        return vector;
      case RENDERPASS_VOLUME_LIGHT:
        return vector;
      default:
        BLI_assert(0);
        return combined;
    }
  }

  /**
   * Iterator
   **/

  struct RenderPassItem {
    Film *&film;
    eRenderPassBit pass_bit;

    constexpr explicit RenderPassItem(Film *&film_, eRenderPassBit pass_bit_)
        : film(film_), pass_bit(pass_bit_){};
  };

  class Iterator {
   private:
    RenderPasses &render_passes_;
    int64_t current_;

   public:
    constexpr explicit Iterator(RenderPasses &rpasses, int64_t current)
        : render_passes_(rpasses), current_(current){};

    constexpr Iterator &operator++()
    {
      while (current_ < RENDERPASS_MAX) {
        current_ <<= 1;
        if (current_ & render_passes_.enabled_passes_) {
          break;
        }
      }
      return *this;
    }

    constexpr friend bool operator!=(const Iterator &a, const Iterator &b)
    {
      return a.current_ != b.current_;
    }

    constexpr RenderPassItem operator*()
    {
      eRenderPassBit pass_bit = static_cast<eRenderPassBit>(current_);
      return RenderPassItem(render_passes_.render_pass_bit_to_film_p(pass_bit), pass_bit);
    }
  };

  /* Iterator over all enabled passes. */
  constexpr Iterator begin()
  {
    return Iterator(*this, 1);
  }

  constexpr Iterator end()
  {
    return Iterator(*this, power_of_2_max_constexpr(RENDERPASS_MAX));
  }
};

/** \} */

}  // namespace blender::eevee
