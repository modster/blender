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

/** \file
 * \ingroup eevee
 *
 * A film is a fullscreen buffer (usually at output extent)
 * that will be able to accumulate sample in any distorted camera_type
 * using a pixel filter.
 *
 * Input needs to be jittered so that the filter converges to the right result.
 */

#pragma once

#include "DRW_render.h"

#include "eevee_camera.hh"
#include "eevee_sampling.hh"
#include "eevee_shader.hh"
#include "eevee_wrapper.hh"

namespace blender::eevee {

class Instance;

/* -------------------------------------------------------------------- */
/** \name Film
 * \{ */

class Film {
 private:
  Instance &inst_;

  /** Owned resources. */
  eevee::Framebuffer read_result_fb_;
  eevee::Framebuffer accumulation_fb_[2];
  eevee::Texture data_tx_[2];
  eevee::Texture weight_tx_[2];
  /** First sample in case we need to blend using it or just reuse it. */
  eevee::Texture first_sample_tx_;

  /** Reference to first_sample_tx_ or data_tx_ depending on the context. */
  GPUTexture *first_sample_ref_;

  // DRWPass *clear_ps_ = nullptr;
  DRWPass *accumulate_ps_ = nullptr;
  DRWPass *resolve_ps_ = nullptr;

  /** Shader parameter, not allocated. */
  GPUTexture *input_tx_;
  /** ViewProjection matrix used to render the input. */
  // float src_persmat_[4][4];
  /** ViewProjection matrix Inverse used to render the input. */
  // float src_persinv_[4][4];

  StructBuffer<FilmData> data_;

  /** True if offset or size changed. */
  bool has_changed_ = true;

  /** Debug static name. */
  StringRefNull name_;

 public:
  /* NOTE: name needs to be static. */
  Film(Instance &inst, eFilmDataType data_type, const char *name) : inst_(inst), name_(name)
  {
    data_.extent[0] = data_.extent[1] = -1;
    data_.data_type = data_type;
    data_.use_history = 0;
  }

  ~Film(){};

  void init(const ivec2 &full_extent, const rcti *output_rect);

  void sync(void);
  void end_sync(void);

  void accumulate(GPUTexture *input, const DRWView *view);

  void resolve_viewport(GPUFrameBuffer *target);

  void read_result(float *data);

 private:
  bool do_smooth_viewport_smooth_transition(void)
  {
    return ELEM(data_.data_type, FILM_DATA_COLOR, FILM_DATA_COLOR_LOG) &&
           !DRW_state_is_image_render();
  }
};

/** \} */

}  // namespace blender::eevee
