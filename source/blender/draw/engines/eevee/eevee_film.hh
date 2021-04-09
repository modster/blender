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

#include "BLI_rect.h"

#include "GPU_framebuffer.h"
#include "GPU_texture.h"

#include "DRW_render.h"

#include "eevee_camera.hh"
#include "eevee_sampling.hh"
#include "eevee_shader.hh"

namespace blender::eevee {

/* -------------------------------------------------------------------- */
/** \name FilmData
 * \{ */

static eGPUTextureFormat to_gpu_texture_format(eFilmDataType film_type)
{
  switch (film_type) {
    default:
    case FILM_DATA_COLOR_LOG:
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

inline bool operator==(const FilmData &a, const FilmData &b)
{
  return equals_v2v2_int(a.extent, b.extent) && equals_v2v2_int(a.offset, b.offset);
}

inline bool operator!=(const FilmData &a, const FilmData &b)
{
  return !(a == b);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Film
 * \{ */

class Film {
 private:
  /** Owned resources. */
  GPUFrameBuffer *read_result_fb_ = nullptr;
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
  Sampling &sampling_;

  /** True if offset or size changed. */
  bool has_changed_ = true;

  /** Debug static name. */
  StringRefNull name_;

 public:
  /* NOTE: name needs to be static. */
  Film(ShaderModule &shaders,
       Camera &camera,
       Sampling &sampling,
       eFilmDataType data_type,
       const char *name)

      : shaders_(shaders), camera_(camera), sampling_(sampling), name_(name)
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
    GPU_FRAMEBUFFER_FREE_SAFE(read_result_fb_);
    for (int i = 0; i < 2; i++) {
      GPU_FRAMEBUFFER_FREE_SAFE(accumulation_fb_[i]);
      GPU_TEXTURE_FREE_SAFE(data_tx_[i]);
      GPU_TEXTURE_FREE_SAFE(weight_tx_[i]);
    }
    data_.use_history = 0;
  }

  void init(const int full_extent[2], const rcti *output_rect)
  {
    FilmData data = data_;
    data.extent[0] = BLI_rcti_size_x(output_rect);
    data.extent[1] = BLI_rcti_size_y(output_rect);
    data.offset[0] = output_rect->xmin;
    data.offset[1] = output_rect->ymin;

    has_changed_ = data_ != data;

    if (has_changed_) {
      data_ = data;
      sampling_.reset();
      this->clear();
    }

    for (int i = 0; i < 2; i++) {
      data_.uv_scale[i] = 1.0f / full_extent[i];
      data_.uv_scale_inv[i] = full_extent[i];
      data_.uv_bias[i] = data_.offset[i] / (float)full_extent[i];
    }
  }

  void sync(void)
  {
    /* TODO reprojection. */

    if (camera_.has_changed()) {
      has_changed_ = true;
      data_.use_history = 0;
    }

    char full_name[32];
    for (int i = 0; i < 2; i++) {
      if (data_tx_[i] == nullptr) {
        eGPUTextureFormat tex_format = to_gpu_texture_format(data_.data_type);
        int *extent = data_.extent;
        SNPRINTF(full_name, "Film.%s.data", name_.c_str());
        data_tx_[i] = GPU_texture_create_2d(full_name, UNPACK2(extent), 1, tex_format, nullptr);
        /* TODO(fclem) The weight texture could be shared between all similar accumulators. */
        SNPRINTF(full_name, "Film.%s.weight", name_.c_str());
        weight_tx_[i] = GPU_texture_create_2d(full_name, UNPACK2(extent), 1, GPU_R16F, nullptr);

        GPU_framebuffer_ensure_config(&accumulation_fb_[i],
                                      {
                                          GPU_ATTACHMENT_NONE,
                                          GPU_ATTACHMENT_TEXTURE(data_tx_[i]),
                                          GPU_ATTACHMENT_TEXTURE(weight_tx_[i]),
                                      });
      }
    }

    eGPUSamplerState no_filter = GPU_SAMPLER_DEFAULT;
    {
      SNPRINTF(full_name, "Film.%s.Accumulate", name_.c_str());
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
      SNPRINTF(full_name, "Film.%s.Resolve", name_.c_str());
      DRWState state = DRW_STATE_WRITE_COLOR | DRW_STATE_WRITE_DEPTH | DRW_STATE_DEPTH_ALWAYS;
      resolve_ps_ = DRW_pass_create(full_name, state);
      GPUShader *sh = shaders_.static_shader_get(FILM_RESOLVE);
      DRWShadingGroup *grp = DRW_shgroup_create(sh, resolve_ps_);
      DRW_shgroup_uniform_block(grp, "film_block", ubo_);
      DRW_shgroup_uniform_texture_ref_ex(grp, "data_tx", &data_tx_[0], no_filter);
      DRW_shgroup_uniform_texture_ref_ex(grp, "weight_tx", &weight_tx_[0], no_filter);
      DRW_shgroup_call_procedural_triangles(grp, NULL, 1);
    }

    if (data_.use_history == 0) {
      GPU_uniformbuf_update(ubo_, &data_);
    }
  }

  void accumulate(GPUTexture *input, const DRWView *view)
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

  void resolve_viewport(GPUFrameBuffer *target)
  {
    int viewport[4];

    GPU_framebuffer_bind(target);
    GPU_framebuffer_viewport_get(target, viewport);

    const bool use_render_border = (data_.offset[0] > 0) || (data_.offset[1] > 0) ||
                                   (data_.extent[0] < viewport[2]) ||
                                   (data_.extent[1] < viewport[3]);
    if (use_render_border) {
      if (has_changed_) {
        /* Film is cropped and does not fill the view completely. Clear the background. */
        if (data_.data_type == FILM_DATA_DEPTH) {
          GPU_framebuffer_clear_depth(target, 1.0f);
        }
        else {
          float color[4] = {0.0f, 0.0f, 0.0f, 0.0f};
          GPU_framebuffer_clear_color(target, color);
        }
      }
      GPU_framebuffer_viewport_set(target, UNPACK2(data_.offset), UNPACK2(data_.extent));
    }

    DRW_draw_pass(resolve_ps_);

    if (use_render_border) {
      GPU_framebuffer_viewport_reset(target);
    }
  }

  void read_result(float *data)
  {
    /* Resolve onto the next data texture. */
    GPU_framebuffer_ensure_config(&read_result_fb_,
                                  {
                                      GPU_ATTACHMENT_NONE,
                                      GPU_ATTACHMENT_TEXTURE(data_tx_[1]),
                                  });
    GPU_framebuffer_bind(read_result_fb_);
    DRW_draw_pass(resolve_ps_);

    eGPUTextureFormat format = to_gpu_texture_format(data_.data_type);
    int channel_count = GPU_texture_component_len(format);
    GPU_framebuffer_read_color(
        read_result_fb_, 0, 0, UNPACK2(data_.extent), channel_count, 0, GPU_DATA_FLOAT, data);
  }
};

/** \} */

}  // namespace blender::eevee
