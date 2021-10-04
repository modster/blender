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

#include "BLI_rect.h"

#include "GPU_framebuffer.h"
#include "GPU_texture.h"

#include "DRW_render.h"

#include "eevee_film.hh"
#include "eevee_instance.hh"

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
    case FILM_DATA_MOTION:
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
  return (a.extent == b.extent) && (a.offset == b.offset);
}

inline bool operator!=(const FilmData &a, const FilmData &b)
{
  return !(a == b);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Film
 * \{ */

void Film::init(const ivec2 &full_extent, const rcti *output_rect)
{
  FilmData data = data_;
  data.extent = ivec2(BLI_rcti_size_x(output_rect), BLI_rcti_size_y(output_rect));
  data.offset = ivec2(output_rect->xmin, output_rect->ymin);

  has_changed_ = data_ != data;

  if (has_changed_) {
    data_ = data;
    inst_.sampling.reset();
  }

  data_.opacity = 1.0f;
  data_.uv_scale = 1.0f / vec2(full_extent);
  data_.uv_scale_inv = full_extent;
  data_.uv_bias = data_.offset / vec2(full_extent);
}

void Film::sync(void)
{
  char full_name[32];
  for (int i = 0; i < 2; i++) {
    if (data_tx_[i] == nullptr) {
      eGPUTextureFormat tex_format = to_gpu_texture_format(data_.data_type);
      SNPRINTF(full_name, "Film.%s.data", name_.c_str());
      data_tx_[i].ensure(full_name, UNPACK2(data_.extent), 1, tex_format);
      /* TODO(fclem) The weight texture could be shared between all similar accumulators. */
      SNPRINTF(full_name, "Film.%s.weight", name_.c_str());
      weight_tx_[i].ensure(full_name, UNPACK2(data_.extent), 1, GPU_R16F);

      accumulation_fb_[i].ensure(GPU_ATTACHMENT_NONE,
                                 GPU_ATTACHMENT_TEXTURE(data_tx_[i]),
                                 GPU_ATTACHMENT_TEXTURE(weight_tx_[i]));
    }
  }

  eGPUSamplerState no_filter = GPU_SAMPLER_DEFAULT;
  {
    SNPRINTF(full_name, "Film.%s.Accumulate", name_.c_str());
    accumulate_ps_ = DRW_pass_create(full_name, DRW_STATE_WRITE_COLOR);
    GPUShader *sh = inst_.shaders.static_shader_get(FILM_FILTER);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, accumulate_ps_);
    DRW_shgroup_uniform_block(grp, "film_block", data_.ubo_get());
    DRW_shgroup_uniform_block(grp, "camera_block", inst_.camera.ubo_get());
    DRW_shgroup_uniform_texture_ref_ex(grp, "input_tx", &input_tx_, no_filter);
    DRW_shgroup_uniform_texture_ref_ex(grp, "data_tx", &data_tx_[0], no_filter);
    DRW_shgroup_uniform_texture_ref_ex(grp, "weight_tx", &weight_tx_[0], no_filter);
    DRW_shgroup_call_procedural_triangles(grp, NULL, 1);
  }
  {
    SNPRINTF(full_name, "Film.%s.Resolve", name_.c_str());
    DRWState state = DRW_STATE_WRITE_COLOR;
    eShaderType sh_type = FILM_RESOLVE;
    if (data_.data_type == FILM_DATA_DEPTH) {
      state = DRW_STATE_WRITE_DEPTH | DRW_STATE_DEPTH_ALWAYS;
      sh_type = FILM_RESOLVE_DEPTH;
    }
    resolve_ps_ = DRW_pass_create(full_name, state);
    GPUShader *sh = inst_.shaders.static_shader_get(sh_type);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, resolve_ps_);
    DRW_shgroup_uniform_block(grp, "film_block", data_.ubo_get());
    DRW_shgroup_uniform_texture_ref_ex(grp, "first_sample_tx", &first_sample_ref_, no_filter);
    DRW_shgroup_uniform_texture_ref_ex(grp, "data_tx", &data_tx_[0], no_filter);
    DRW_shgroup_uniform_texture_ref_ex(grp, "weight_tx", &weight_tx_[0], no_filter);
    DRW_shgroup_call_procedural_triangles(grp, NULL, 1);
  }
}

void Film::end_sync()
{
  /* TODO reprojection. */
  if (inst_.sampling.is_reset()) {
    data_.use_history = 0;
  }

  if (inst_.is_viewport()) {
    data_.opacity = inst_.sampling.viewport_smoothing_opacity_factor_get();
  }

  if (data_.use_history == 0 || inst_.is_viewport()) {
    data_.push_update();
  }

  const bool is_first_sample = (inst_.sampling.sample_get() == 1);
  if (do_smooth_viewport_smooth_transition() && (data_.opacity < 1.0f || is_first_sample)) {
    char full_name[32];
    SNPRINTF(full_name, "Film.%s.first_sample", name_.c_str());
    GPUTexture *dtxl_color = DRW_viewport_texture_list_get()->color;
    eGPUTextureFormat tex_format = GPU_texture_format(dtxl_color);
    int extent[2] = {GPU_texture_width(dtxl_color), GPU_texture_height(dtxl_color)};
    first_sample_tx_.ensure(full_name, UNPACK2(extent), 1, tex_format);
    first_sample_ref_ = first_sample_tx_;
  }
  else {
    /* Reuse the data_tx since there is no need to blend. */
    first_sample_tx_.release();
    first_sample_ref_ = data_tx_[0];
  }
}

void Film::accumulate(GPUTexture *input, const DRWView *view)
{
  input_tx_ = input;

  DRW_view_set_active(view);

  GPU_framebuffer_bind(accumulation_fb_[1]);
  DRW_draw_pass(accumulate_ps_);

  SWAP(Framebuffer, accumulation_fb_[0], accumulation_fb_[1]);
  SWAP(Texture, data_tx_[0], data_tx_[1]);
  SWAP(Texture, weight_tx_[0], weight_tx_[1]);

  /* Use history after first sample. */
  if (data_.use_history == 0) {
    data_.use_history = 1;
    data_.push_update();
  }
}

void Film::resolve_viewport(GPUFrameBuffer *target)
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

  /* Minus one because we already incremented it in step() which is the first
   * thing to happen in the sample loop. */
  const bool is_first_sample = (inst_.sampling.sample_get() - 1 == 1);
  const bool is_only_one_sample = is_first_sample && inst_.sampling.finished();
  if (is_first_sample && !is_only_one_sample && do_smooth_viewport_smooth_transition()) {
    DefaultTextureList *dtxl = DRW_viewport_texture_list_get();
    GPU_texture_copy(first_sample_tx_, dtxl->color);
  }

  if (use_render_border) {
    GPU_framebuffer_viewport_reset(target);
  }
}

void Film::read_result(float *data)
{
  /* Resolve onto the next data texture. */
  read_result_fb_.ensure(GPU_ATTACHMENT_NONE, GPU_ATTACHMENT_TEXTURE(data_tx_[1]));
  GPU_framebuffer_bind(read_result_fb_);
  DRW_draw_pass(resolve_ps_);

  eGPUTextureFormat format = to_gpu_texture_format(data_.data_type);
  int channel_count = GPU_texture_component_len(format);
  GPU_framebuffer_read_color(
      read_result_fb_, 0, 0, UNPACK2(data_.extent), channel_count, 0, GPU_DATA_FLOAT, data);
}

/** \} */

}  // namespace blender::eevee
