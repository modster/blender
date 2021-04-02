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

#include "GPU_framebuffer.h"
#include "GPU_texture.h"

#include "DRW_render.h"

#include "eevee_camera.hh"
#include "eevee_shader.hh"

namespace blender::eevee {

static eGPUTextureFormat to_gpu_texture_format(eFilmDataType film_type)
{
  switch (film_type) {
    default:
    case FILM_DATA_COLOR:
    case FILM_DATA_VEC4:
      return GPU_RGBA16F;
    case FILM_DATA_FLOAT:
      return GPU_R16F;
    case FILM_DATA_VEC2:
      return GPU_RG16F;
    case FILM_DATA_NORMAL:
      return GPU_RGB10_A2;
    case FILM_DATA_DEPTH:
      return GPU_R32F;
  }
}

typedef struct Film {
 private:
  /** Owned resources. */
  GPUFrameBuffer *accumulation_fb_[2] = {nullptr};
  GPUTexture *data_tx_[2] = {nullptr};
  GPUTexture *weight_tx_[2] = {nullptr};

  DRWPass *clear_ps_ = nullptr;
  DRWPass *accumulate_ps_ = nullptr;
  DRWPass *resolve_ps_ = nullptr;

  /** Shader parameter, not allocated. */
  GPUTexture *input_tx_;
  /** ViewProjection matrix used to render the input. */
  float src_persmat_[4][4];
  /** ViewProjection matrix Inverse used to render the input. */
  float src_persinv_[4][4];

  FilmData data_;
  GPUUniformBuf *ubo_ = nullptr;

  ShaderModule &shaders_;
  Camera &camera_;

  /** Debug static name. */
  const char *name_;

 public:
  /* NOTE: name needs to be static. */
  Film(ShaderModule &shaders, Camera &camera, eFilmDataType data_type, const char *name)

      : shaders_(shaders), camera_(camera), name_(name)
  {
    data_.extent[0] = data_.extent[1] = -1;
    data_.data_type = data_type;
    data_.use_history = 0;
    ubo_ = GPU_uniformbuf_create_ex(sizeof(FilmData), nullptr, "FilmData");
  }

  ~Film()
  {
    this->clear();
    DRW_UBO_FREE_SAFE(ubo_);
  }

  void clear(void)
  {
    for (int i = 0; i < 2; i++) {
      GPU_FRAMEBUFFER_FREE_SAFE(accumulation_fb_[i]);
      GPU_TEXTURE_FREE_SAFE(data_tx_[i]);
      GPU_TEXTURE_FREE_SAFE(weight_tx_[i]);
    }
    data_.use_history = 0;
  }

  void init(const int extent[2])
  {
    char full_name[32];

    /* TODO reprojection. */
    data_.use_history = camera_.has_changed() ? 0 : 1;

    if (!equals_v2v2_int(data_.extent, extent)) {
      copy_v2_v2_int(data_.extent, extent);
      this->clear();
    }

    for (int i = 0; i < 2; i++) {
      if (data_tx_[i] == nullptr) {
        eGPUTextureFormat tex_format = to_gpu_texture_format(data_.data_type);
        SNPRINTF(full_name, "Film.%s.data", name_);
        data_tx_[i] = GPU_texture_create_2d(full_name, UNPACK2(extent), 1, tex_format, nullptr);
        /* TODO(fclem) The weight texture could be shared between all similar accumulators. */
        SNPRINTF(full_name, "Film.%s.weight", name_);
        weight_tx_[i] = GPU_texture_create_2d(full_name, UNPACK2(extent), 1, GPU_R16F, nullptr);

        GPU_framebuffer_ensure_config(&accumulation_fb_[i],
                                      {
                                          GPU_ATTACHMENT_NONE,
                                          GPU_ATTACHMENT_TEXTURE(data_tx_[i]),
                                          GPU_ATTACHMENT_TEXTURE(weight_tx_[i]),
                                      });
      }
    }

    if (data_.use_history == 0) {
      GPU_uniformbuf_update(ubo_, &data_);
    }

    eGPUSamplerState no_filter = GPU_SAMPLER_DEFAULT;
    {
      SNPRINTF(full_name, "Film.%s.Accumulate", name_);
      accumulate_ps_ = DRW_pass_create(full_name, DRW_STATE_WRITE_COLOR);
      GPUShader *sh = shaders_.static_shader_get(FILM_FILTER);
      DRWShadingGroup *grp = DRW_shgroup_create(sh, accumulate_ps_);
      DRW_shgroup_uniform_block(grp, "film_block", ubo_);
      DRW_shgroup_uniform_block(grp, "camera_block", camera_.ubo_get());
      DRW_shgroup_uniform_texture_ref_ex(grp, "input_tx", &input_tx_, no_filter);
      DRW_shgroup_uniform_texture_ref_ex(grp, "data_tx", &data_tx_[0], no_filter);
      DRW_shgroup_uniform_texture_ref_ex(grp, "weight_tx", &weight_tx_[0], no_filter);
      DRW_shgroup_call_procedural_triangles(grp, NULL, 1);
    }
    {
      SNPRINTF(full_name, "Film.%s.Resolve", name_);
      DRWState state = DRW_STATE_WRITE_COLOR | DRW_STATE_WRITE_DEPTH | DRW_STATE_DEPTH_ALWAYS;
      resolve_ps_ = DRW_pass_create(full_name, state);
      GPUShader *sh = shaders_.static_shader_get(FILM_RESOLVE);
      DRWShadingGroup *grp = DRW_shgroup_create(sh, resolve_ps_);
      DRW_shgroup_uniform_block(grp, "film_block", ubo_);
      DRW_shgroup_uniform_texture_ref_ex(grp, "data_tx", &data_tx_[0], no_filter);
      DRW_shgroup_uniform_texture_ref_ex(grp, "weight_tx", &weight_tx_[0], no_filter);
      DRW_shgroup_call_procedural_triangles(grp, NULL, 1);
    }
  }

  void accumulate(GPUTexture *input, DRWView *view)
  {
    input_tx_ = input;

    DRW_view_set_active(view);

    GPU_framebuffer_bind(accumulation_fb_[1]);
    DRW_draw_pass(accumulate_ps_);

    SWAP(GPUFrameBuffer *, accumulation_fb_[0], accumulation_fb_[1]);
    SWAP(GPUTexture *, data_tx_[0], data_tx_[1]);
    SWAP(GPUTexture *, weight_tx_[0], weight_tx_[1]);

    /* Use history after first sample. */
    if (data_.use_history == 0) {
      data_.use_history = 1;
      GPU_uniformbuf_update(ubo_, &data_);
    }
  }

  void resolve_onto(GPUFrameBuffer *target)
  {
    GPU_framebuffer_bind(target);

    DRW_draw_pass(resolve_ps_);
  }

  void read_to_memory(float *data)
  {
    /* TODO(fclem) implement. */
    (void)data;
  }
} Film;

}  // namespace blender::eevee
