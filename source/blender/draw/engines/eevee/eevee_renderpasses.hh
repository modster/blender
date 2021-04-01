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

#include "eevee_film.hh"

namespace blender::eevee {

enum eRenderPassBit {
  RENDERPASS_NONE = 0,
  RENDERPASS_COMBINED = (1 << 0),
  RENDERPASS_DEPTH = (1 << 1),
  RENDERPASS_NORMAL = (1 << 2),
};

ENUM_OPERATORS(eRenderPassBit, RENDERPASS_NORMAL)

typedef struct RenderPasses {
 public:
  Film *combined = nullptr;
  Film *depth = nullptr;
  Film *normal = nullptr;
  Vector<Film *> aovs;

 private:
  ShaderModule &shaders_;
  Camera &camera_;
  eRenderPassBit enabled_passes_ = RENDERPASS_NONE;

  int extent_[2];

 public:
  RenderPasses(ShaderModule &shaders, Camera &camera) : shaders_(shaders), camera_(camera){};

  ~RenderPasses()
  {
    delete combined;
    delete depth;
    delete normal;
  }

  void configure(eRenderPassBit passes, const int extent[2])
  {
    copy_v2_v2_int(extent_, extent);
    enabled_passes_ = passes;

    pass_configure(passes, RENDERPASS_COMBINED, combined, FILM_DATA_COLOR, "Combined");
    pass_configure(passes, RENDERPASS_DEPTH, depth, FILM_DATA_DEPTH, "Depth");
    pass_configure(passes, RENDERPASS_NORMAL, normal, FILM_DATA_NORMAL, "Normal");
  }

  void init(void)
  {
    if (combined) {
      combined->init(extent_);
    }
    if (depth) {
      depth->init(extent_);
    }
    if (normal) {
      normal->init(extent_);
    }
    for (Film *aov : aovs) {
      aov->init(extent_);
    }
  }

  eRenderPassBit enabled_passes_get(void)
  {
    return enabled_passes_;
  }

 private:
  inline void pass_configure(eRenderPassBit passes,
                             eRenderPassBit pass_bit,
                             Film *&pass,
                             eFilmDataType type,
                             const char *name)
  {
    bool enable = (passes & pass_bit) != 0;
    if (enable && pass == nullptr) {
      pass = new Film(shaders_, camera_, type, name);
    }
    else if (!enable && pass != nullptr) {
      /* Delete unused passes. */
      delete pass;
      pass = nullptr;
    }
  }

} RenderPasses;

}  // namespace blender::eevee
