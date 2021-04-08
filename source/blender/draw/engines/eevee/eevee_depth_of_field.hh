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
 * Depth of field post process effect.
 *
 * There are 2 methods to achieve this effect.
 * - The first uses projection matrix offsetting and sample accumulation to give
 * reference quality depth of field. But this needs many samples to hide the
 * under-sampling.
 * - The second one is a post-processing based one. It follows the
 * implementation described in the presentation "Life of a Bokeh - Siggraph
 * 2018" from Guillaume Abadie. There are some difference with our actual
 * implementation that prioritize quality.
 */

#include "GPU_uniform_buffer.h"

#include "eevee_camera.hh"
#include "eevee_sampling.hh"
#include "eevee_shader.hh"
#include "eevee_shader_shared.hh"

namespace blender::eevee {

/* -------------------------------------------------------------------- */
/** \name Depth of field
 * \{ */

class DepthOfField {
 private:
  ShaderModule &shaders_;
  Sampling &sampling_;

  DepthOfFieldData data_;
  GPUUniformBuf *ubo_;

  /** Textures from pool. Not owned. */
  GPUTexture *bokeh_gather_lut_tx_ = nullptr;
  GPUTexture *bokeh_resolve_lut_tx_ = nullptr;
  GPUTexture *bokeh_scatter_lut_tx_ = nullptr;
  GPUTexture *color_bg_tx_ = nullptr;
  GPUTexture *color_fg_tx_ = nullptr;
  GPUTexture *color_holefill_tx_ = nullptr;
  GPUTexture *occlusion_tx_ = nullptr;
  GPUTexture *reduce_downsample_tx_ = nullptr;
  GPUTexture *scatter_src_tx_ = nullptr;
  GPUTexture *setup_coc_tx_ = nullptr;
  GPUTexture *setup_color_tx_ = nullptr;
  GPUTexture *tiles_bg_tx_ = nullptr;
  GPUTexture *tiles_fg_tx_ = nullptr;
  GPUTexture *tiles_dilated_bg_tx_ = nullptr;
  GPUTexture *tiles_dilated_fg_tx_ = nullptr;
  GPUTexture *weight_bg_tx_ = nullptr;
  GPUTexture *weight_fg_tx_ = nullptr;
  GPUTexture *weight_holefill_tx_ = nullptr;
  /** Allocated textures. Owned. */
  GPUTexture *reduced_coc_tx_ = nullptr;
  GPUTexture *reduced_color_tx_ = nullptr;
  /** Input texture. Not owned. */
  GPUTexture *input_color_tx_ = nullptr;
  GPUTexture *input_depth_tx_ = nullptr;
  /** Passes. Not owned. */
  DRWPass *bokeh_lut_ps_ = nullptr;
  DRWPass *gather_bg_ps_ = nullptr;
  DRWPass *gather_fg_ps_ = nullptr;
  DRWPass *filter_ps_ = nullptr;
  DRWPass *gather_holefill_ps_ = nullptr;
  DRWPass *reduce_copy_ps_ = nullptr;
  DRWPass *reduce_downsample_ps_ = nullptr;
  DRWPass *reduce_recursive_ps_ = nullptr;
  DRWPass *resolve_ps_ = nullptr;
  DRWPass *scatter_bg_ps_ = nullptr;
  DRWPass *scatter_fg_ps_ = nullptr;
  DRWPass *setup_ps_ = nullptr;
  DRWPass *tiles_dilate_minabs_ps_ = nullptr;
  DRWPass *tiles_dilate_minmax_ps_ = nullptr;
  DRWPass *tiles_flatten_ps_ = nullptr;
  /** Framebuffers. Owned.  */
  GPUFrameBuffer *bokeh_lut_fb_ = nullptr;
  GPUFrameBuffer *filter_bg_fb_ = nullptr;
  GPUFrameBuffer *filter_fg_fb_ = nullptr;
  GPUFrameBuffer *gather_fb_ = nullptr;
  GPUFrameBuffer *gather_filter_bg_fb_ = nullptr;
  GPUFrameBuffer *gather_holefill_fb_ = nullptr;
  GPUFrameBuffer *reduce_copy_fb_ = nullptr;
  GPUFrameBuffer *reduce_downsample_fb_ = nullptr;
  GPUFrameBuffer *reduce_fb_ = nullptr;
  GPUFrameBuffer *resolve_fb_ = nullptr;
  GPUFrameBuffer *scatter_bg_fb_ = nullptr;
  GPUFrameBuffer *scatter_fg_fb_ = nullptr;
  GPUFrameBuffer *setup_fb_ = nullptr;
  GPUFrameBuffer *tiles_dilate_fb_ = nullptr;
  GPUFrameBuffer *tiles_flatten_fb_ = nullptr;

  /** Scene settings that are immutable. */
  float user_overblur_;
  float fx_max_coc_;
  /** Use Hiqh Quality (expensive) in-focus gather pass. */
  bool do_hq_slight_focus_;
  /** Use jittered depth of field where we randomize camera location. */
  bool do_jitter_;

  /** Circle of Confusion radius for FX DoF passes. Is in view X direction in [0..1] range. */
  float fx_radius_;
  /** Circle of Confusion radius for jittered DoF. Is in view X direction in [0..1] range. */
  float jitter_radius_;
  /** Focus distance in view space. */
  float focus_distance_;
  /** Extent of the input buffer. */
  int extent_[2];

  /** Tile dilation uniforms. */
  int tiles_dilate_slight_focus_;
  int tiles_dilate_ring_count_;
  int tiles_dilate_ring_width_multiplier_;

  /** Reduce pass info. */
  int reduce_steps_;

  /** Static srting pointer. Used as debug name and as UUID for texture pool. */
  StringRefNull view_name_;

 public:
  DepthOfField(ShaderModule &shaders, Sampling &sampling, StringRefNull view_name)
      : shaders_(shaders), sampling_(sampling), view_name_(view_name)
  {
    ubo_ = GPU_uniformbuf_create_ex(sizeof(DepthOfFieldData), nullptr, "DepthOfFieldData");
  };

  ~DepthOfField()
  {
    GPU_TEXTURE_FREE_SAFE(reduced_coc_tx_);
    GPU_TEXTURE_FREE_SAFE(reduced_color_tx_);
    GPU_FRAMEBUFFER_FREE_SAFE(bokeh_lut_fb_);
    GPU_FRAMEBUFFER_FREE_SAFE(filter_bg_fb_);
    GPU_FRAMEBUFFER_FREE_SAFE(filter_fg_fb_);
    GPU_FRAMEBUFFER_FREE_SAFE(gather_fb_);
    GPU_FRAMEBUFFER_FREE_SAFE(gather_filter_bg_fb_);
    GPU_FRAMEBUFFER_FREE_SAFE(gather_holefill_fb_);
    GPU_FRAMEBUFFER_FREE_SAFE(reduce_copy_fb_);
    GPU_FRAMEBUFFER_FREE_SAFE(reduce_downsample_fb_);
    GPU_FRAMEBUFFER_FREE_SAFE(reduce_fb_);
    GPU_FRAMEBUFFER_FREE_SAFE(resolve_fb_);
    GPU_FRAMEBUFFER_FREE_SAFE(scatter_bg_fb_);
    GPU_FRAMEBUFFER_FREE_SAFE(scatter_fg_fb_);
    GPU_FRAMEBUFFER_FREE_SAFE(setup_fb_);
    GPU_FRAMEBUFFER_FREE_SAFE(tiles_dilate_fb_);
    GPU_FRAMEBUFFER_FREE_SAFE(tiles_flatten_fb_);
    DRW_UBO_FREE_SAFE(ubo_);
  };

  void init(const Scene *scene)
  {
    const SceneEEVEE &sce_eevee = scene->eevee;
    do_hq_slight_focus_ = (sce_eevee.flag & SCE_EEVEE_DOF_HQ_SLIGHT_FOCUS) != 0;
    do_jitter_ = (sce_eevee.flag & SCE_EEVEE_DOF_JITTER) != 0;
    user_overblur_ = sce_eevee.bokeh_overblur / 100.0f;
    fx_max_coc_ = sce_eevee.bokeh_max_size;
    data_.scatter_color_threshold = sce_eevee.bokeh_threshold;
    data_.scatter_neighbor_max_color = sce_eevee.bokeh_neighbor_max;
    data_.denoise_factor = sce_eevee.bokeh_denoise_fac;
    /* Default to no depth of field. */
    fx_radius_ = 0.0f;
    jitter_radius_ = 0.0f;
  }

  void sync(const Camera &camera, const float winmat[4][4], int input_extent[2])
  {
    const Object *camera_object_eval = camera.blender_camera_get();
    const ::Camera *cam = (camera_object_eval) ?
                              reinterpret_cast<const ::Camera *>(camera_object_eval->data) :
                              nullptr;

    if (cam == nullptr || (cam->dof.flag & CAM_DOF_ENABLED) == 0) {
      fx_radius_ = 0.0f;
      jitter_radius_ = 0.0f;
      return;
    }

    copy_v2_v2_int(extent_, input_extent);

    data_.bokeh_blades = cam->dof.aperture_blades;
    data_.bokeh_rotation = cam->dof.aperture_rotation;
    data_.bokeh_anisotropic_scale[0] = clamp_f(1.0f / cam->dof.aperture_ratio, 0.00001f, 1.0f);
    data_.bokeh_anisotropic_scale[1] = clamp_f(cam->dof.aperture_ratio, 0.00001f, 1.0f);
    copy_v2_v2(data_.bokeh_anisotropic_scale_inv, data_.bokeh_anisotropic_scale);
    invert_v2(data_.bokeh_anisotropic_scale_inv);

    focus_distance_ = BKE_camera_object_dof_distance(camera_object_eval);
    float fstop = max_ff(cam->dof.aperture_fstop, 1e-5f);
    float aperture = 1.0f / (2.0f * fstop);
    if (cam->type == CAM_PERSP) {
      aperture *= cam->lens * 1e-3f;
    }

    if (cam->type == CAM_ORTHO) {
      /* FIXME: Why is this needed? Some kind of implicit unit conversion? */
      aperture *= 0.015f;
      /* Really strange behavior from Cycles but replicating. */
      focus_distance_ += cam->clip_start;
    }

    if (cam->type == CAM_PANO) {
      /* FIXME: Eyeballed. */
      aperture *= 0.185f;
    }

    if (cam->dof.aperture_ratio < 1.0) {
      /* If ratio is scaling the bokeh outwards, we scale the aperture so that
       * the gather kernel size will encompass the maximum axis. */
      aperture /= max_ff(cam->dof.aperture_ratio, 1e-5f);
    }

    /* Balance blur radius between fx dof and jitter dof. */
    if (do_jitter_ && (sampling_.dof_ring_count_get() > 0) && (cam->type != CAM_PANO)) {
      /* Compute a minimal overblur radius to fill the gaps between the samples.
       * This is just the simplified form of dividing the area of the bokeh by
       * the number of samples. */
      float minimal_overblur = 1.0f / sqrtf(sampling_.dof_sample_count_get());

      fx_radius_ = (minimal_overblur + user_overblur_) * aperture;
      /* Avoid dilating the shape. Over-blur only soften. */
      jitter_radius_ = max_ff(0.0f, aperture - fx_radius_);
    }
    else {
      jitter_radius_ = 0.0f;
      fx_radius_ = aperture;
    }

    if (fx_max_coc_ > 0.0f && fx_radius_ > 0.0f) {
      data_.camera_type = camera.data_get().type;
      /* OPTI(fclem) Could be optimized. */
      float jitter[3] = {fx_radius_, 0.0f, -focus_distance_};
      float center[3] = {0.0f, 0.0f, -focus_distance_};
      mul_project_m4_v3(winmat, jitter);
      mul_project_m4_v3(winmat, center);
      /* Simplify CoC calculation to a simple MADD. */
      if (data_.camera_type != CAMERA_ORTHO) {
        data_.coc_bias = -(center[0] - jitter[0]) * 0.5f * extent_[0];
        data_.coc_mul = focus_distance_ * data_.coc_bias;
      }
      else {
        data_.coc_mul = (center[0] - jitter[0]) * 0.5f * extent_[0];
        data_.coc_bias = focus_distance_ * data_.coc_mul;
      }

      float min_fg_coc = coc_radius_from_camera_depth(data_, -cam->clip_start);
      float max_bg_coc = coc_radius_from_camera_depth(data_, -cam->clip_end);
      if (data_.camera_type != CAMERA_ORTHO) {
        /* Background is at infinity so maximum CoC is the limit of the function at -inf. */
        /* NOTE: we only do this for perspective camera since orthographic coc limit is inf. */
        max_bg_coc = data_.coc_bias;
      }
      /* Clamp with user defined max. */
      data_.coc_abs_max = min_ff(max_ff(fabsf(min_fg_coc), fabsf(max_bg_coc)), fx_max_coc_);

      bokeh_lut_pass_sync();
      setup_pass_sync();
      tiles_prepare_pass_sync();
      reduce_pass_sync();
      convolve_pass_sync();
      resolve_pass_sync();

      GPU_uniformbuf_update(ubo_, &data_);
    }
  }

  void jitter_apply(float winmat[4][4], float viewmat[4][4])
  {
    if (jitter_radius_ == 0.0f) {
      return;
    }
    float radius, theta;
    sampling_.dof_disk_sample_get(&radius, &theta);

    if (data_.bokeh_blades >= 3.0f) {
      theta = circle_to_polygon_angle(data_.bokeh_blades, theta);
      radius *= circle_to_polygon_radius(data_.bokeh_blades, theta);
    }
    radius *= jitter_radius_;
    theta += data_.bokeh_rotation;

    /* Sample in View Space. */
    float sample[2] = {radius * cosf(theta), radius * sinf(theta)};
    mul_v2_v2(sample, data_.bokeh_anisotropic_scale);
    /* Convert to NDC Space. */
    float jitter[3] = {UNPACK2(sample), -focus_distance_};
    float center[3] = {0.0f, 0.0f, -focus_distance_};
    mul_project_m4_v3(winmat, jitter);
    mul_project_m4_v3(winmat, center);

    const bool is_ortho = (winmat[2][3] != -1.0f);
    if (is_ortho) {
      mul_v2_fl(sample, focus_distance_);
    }
    /* Translate origin. */
    sub_v2_v2(viewmat[3], sample);
    /* Skew winmat Z axis. */
    sub_v2_v2v2(jitter, center, jitter);
    add_v2_v2(winmat[2], jitter);
  }

  /** Will swap input and output texture if rendering happens. The actual output of this function
   * is in intput_tx. */
  void render(GPUTexture *depth_tx, GPUTexture **input_tx, GPUTexture **output_tx)
  {
    if (fx_radius_ == 0.0f || fx_max_coc_ < 0.5f) {
      return;
    }

    input_color_tx_ = *input_tx;
    input_depth_tx_ = depth_tx;

    DRW_stats_group_start("Depth of Field");

    bokeh_lut_pass_render();

    setup_pass_render();
    tiles_prepare_pass_render();
    reduce_pass_render();
    convolve_pass_render();
    resolve_pass_render(*output_tx);

    DRW_stats_group_end();

    /* Swap buffers so that next effect has the right input. */
    *input_tx = *output_tx;
    *output_tx = input_color_tx_;
  }

 private:
  /**
   * Creates bokeh texture.
   **/
  void bokeh_lut_pass_sync(void)
  {
    const bool has_anisotropy = equals_v2v2(data_.bokeh_anisotropic_scale,
                                            data_.bokeh_anisotropic_scale_inv);
    if (has_anisotropy && (data_.bokeh_blades == 0.0)) {
      bokeh_gather_lut_tx_ = nullptr;
      bokeh_scatter_lut_tx_ = nullptr;
      bokeh_resolve_lut_tx_ = nullptr;
      bokeh_lut_ps_ = nullptr;
      return;
    }

    DrawEngineType *owner = (DrawEngineType *)view_name_.c_str();
    int res[2] = {DOF_BOKEH_LUT_SIZE, DOF_BOKEH_LUT_SIZE};

    DRW_PASS_CREATE(bokeh_lut_ps_, DRW_STATE_WRITE_COLOR);
    GPUShader *sh = shaders_.static_shader_get(DOF_BOKEH_LUT);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, bokeh_lut_ps_);
    DRW_shgroup_uniform_block(grp, "dof_block", ubo_);
    DRW_shgroup_call_procedural_triangles(grp, nullptr, 1);

    bokeh_gather_lut_tx_ = DRW_texture_pool_query_2d(UNPACK2(res), GPU_RG16F, owner);
    bokeh_scatter_lut_tx_ = DRW_texture_pool_query_2d(UNPACK2(res), GPU_R16F, owner);
    bokeh_resolve_lut_tx_ = DRW_texture_pool_query_2d(UNPACK2(res), GPU_R16F, owner);

    GPU_framebuffer_ensure_config(&bokeh_lut_fb_,
                                  {
                                      GPU_ATTACHMENT_NONE,
                                      GPU_ATTACHMENT_TEXTURE(bokeh_gather_lut_tx_),
                                      GPU_ATTACHMENT_TEXTURE(bokeh_scatter_lut_tx_),
                                      GPU_ATTACHMENT_TEXTURE(bokeh_resolve_lut_tx_),
                                  });
  }

  void bokeh_lut_pass_render(void)
  {
    if (bokeh_lut_ps_) {
      GPU_framebuffer_bind(bokeh_lut_fb_);
      DRW_draw_pass(bokeh_lut_ps_);
    }
  }

  /**
   * Outputs halfResColorBuffer and halfResCocBuffer.
   **/
  void setup_pass_sync(void)
  {
    DrawEngineType *owner = (DrawEngineType *)view_name_.c_str();
    uint res[2] = {divide_ceil_u(extent_[0], 2), divide_ceil_u(extent_[1], 2)};

    DRW_PASS_CREATE(setup_ps_, DRW_STATE_WRITE_COLOR);
    GPUShader *sh = shaders_.static_shader_get(DOF_SETUP);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, setup_ps_);

    eGPUSamplerState no_filter = GPU_SAMPLER_DEFAULT;
    DRW_shgroup_uniform_texture_ref_ex(grp, "color_tx", &input_color_tx_, no_filter);
    DRW_shgroup_uniform_texture_ref_ex(grp, "depth_tx", &input_depth_tx_, no_filter);
    DRW_shgroup_uniform_block(grp, "dof_block", ubo_);
    DRW_shgroup_call_procedural_triangles(grp, nullptr, 1);

    setup_color_tx_ = DRW_texture_pool_query_2d(UNPACK2(res), GPU_RGBA16F, owner);
    setup_coc_tx_ = DRW_texture_pool_query_2d(UNPACK2(res), GPU_RG16F, owner);

    GPU_framebuffer_ensure_config(&setup_fb_,
                                  {
                                      GPU_ATTACHMENT_NONE,
                                      GPU_ATTACHMENT_TEXTURE(setup_color_tx_),
                                      GPU_ATTACHMENT_TEXTURE(setup_coc_tx_),
                                  });
  }

  void setup_pass_render(void)
  {
    GPU_framebuffer_bind(setup_fb_);
    DRW_draw_pass(setup_ps_);
  }

  /**
   * Outputs min & max COC in each 8x8 half res pixel tiles (so 1/16th of full resolution).
   * Then dilates the min & max CoCs to cover maximum COC values.
   **/
  void tiles_prepare_pass_sync(void)
  {
    DrawEngineType *owner = (DrawEngineType *)view_name_.c_str();
    uint res[2] = {divide_ceil_u(extent_[0], DOF_TILE_DIVISOR),
                   divide_ceil_u(extent_[1], DOF_TILE_DIVISOR)};
    /* WARNING: If you change this, make sure dof_tile_* GLSL constants can be properly encoded. */
    eGPUTextureFormat fg_tile_format = GPU_RGBA16F;
    eGPUTextureFormat bg_tile_format = GPU_R11F_G11F_B10F;

    eGPUSamplerState no_filter = GPU_SAMPLER_DEFAULT;
    {
      DRW_PASS_CREATE(tiles_flatten_ps_, DRW_STATE_WRITE_COLOR);

      GPUShader *sh = shaders_.static_shader_get(DOF_TILES_FLATTEN);
      DRWShadingGroup *grp = DRW_shgroup_create(sh, tiles_flatten_ps_);
      DRW_shgroup_uniform_texture_ref_ex(grp, "coc_tx", &setup_coc_tx_, no_filter);
      DRW_shgroup_call_procedural_triangles(grp, nullptr, 1);

      tiles_fg_tx_ = DRW_texture_pool_query_2d(UNPACK2(res), fg_tile_format, owner);
      tiles_bg_tx_ = DRW_texture_pool_query_2d(UNPACK2(res), bg_tile_format, owner);

      GPU_framebuffer_ensure_config(&tiles_flatten_fb_,
                                    {
                                        GPU_ATTACHMENT_NONE,
                                        GPU_ATTACHMENT_TEXTURE(tiles_fg_tx_),
                                        GPU_ATTACHMENT_TEXTURE(tiles_bg_tx_),
                                    });
    }
    {
      DRW_PASS_CREATE(tiles_dilate_minmax_ps_, DRW_STATE_WRITE_COLOR);
      DRW_PASS_CREATE(tiles_dilate_minabs_ps_, DRW_STATE_WRITE_COLOR);

      for (int pass = 0; pass < 2; pass++) {
        DRWPass *drw_pass = (pass == 0) ? tiles_dilate_minmax_ps_ : tiles_dilate_minabs_ps_;
        GPUShader *sh = shaders_.static_shader_get((pass == 0) ? DOF_TILES_DILATE_MINMAX :
                                                                 DOF_TILES_DILATE_MINABS);
        DRWShadingGroup *grp = DRW_shgroup_create(sh, drw_pass);
        DRW_shgroup_uniform_texture_ref_ex(grp, "tiles_fg_tx", &tiles_fg_tx_, no_filter);
        DRW_shgroup_uniform_texture_ref_ex(grp, "tiles_bg_tx", &tiles_bg_tx_, no_filter);
        DRW_shgroup_uniform_bool(grp, "dilate_slight_focus", &tiles_dilate_slight_focus_, 1);
        DRW_shgroup_uniform_int(grp, "ring_count", &tiles_dilate_ring_count_, 1);
        DRW_shgroup_uniform_int(
            grp, "ring_width_multiplier", &tiles_dilate_ring_width_multiplier_, 1);
        DRW_shgroup_call_procedural_triangles(grp, nullptr, 1);
      }

      tiles_dilated_fg_tx_ = DRW_texture_pool_query_2d(UNPACK2(res), fg_tile_format, owner);
      tiles_dilated_bg_tx_ = DRW_texture_pool_query_2d(UNPACK2(res), bg_tile_format, owner);

      GPU_framebuffer_ensure_config(&tiles_dilate_fb_,
                                    {
                                        GPU_ATTACHMENT_NONE,
                                        GPU_ATTACHMENT_TEXTURE(tiles_dilated_fg_tx_),
                                        GPU_ATTACHMENT_TEXTURE(tiles_dilated_bg_tx_),
                                    });
    }
  }

  void tiles_prepare_pass_render(void)
  {
    GPU_framebuffer_bind(tiles_flatten_fb_);
    DRW_draw_pass(tiles_flatten_ps_);

    /* Run dilation twice. One for minmax and one for minabs. */
    for (int pass = 0; pass < 2; pass++) {
      /* Error introduced by gather center jittering. */
      const float error_multiplier = 1.0f + 1.0f / (DOF_GATHER_RING_COUNT + 0.5f);
      int dilation_end_radius = ceilf((fx_max_coc_ * error_multiplier) / DOF_TILE_DIVISOR);

      /* This algorithm produce the exact dilation radius by dividing it in multiple passes. */
      int dilation_radius = 0;
      while (dilation_radius < dilation_end_radius) {
        /* Dilate slight focus only on first iteration. */
        tiles_dilate_slight_focus_ = (dilation_radius == 0) ? 1 : 0;

        int remainder = dilation_end_radius - dilation_radius;
        /* Do not step over any unvisited tile. */
        int max_multiplier = dilation_radius + 1;

        int ring_count = min_ii(DOF_DILATE_RING_COUNT, ceilf(remainder / (float)max_multiplier));
        int multiplier = min_ii(max_multiplier, floor(remainder / (float)ring_count));

        dilation_radius += ring_count * multiplier;

        tiles_dilate_ring_count_ = ring_count;
        tiles_dilate_ring_width_multiplier_ = multiplier;

        GPU_framebuffer_bind(tiles_dilate_fb_);
        DRW_draw_pass((pass == 0) ? tiles_dilate_minmax_ps_ : tiles_dilate_minabs_ps_);

        SWAP(GPUFrameBuffer *, tiles_dilate_fb_, tiles_flatten_fb_);
        SWAP(GPUTexture *, tiles_dilated_bg_tx_, tiles_bg_tx_);
        SWAP(GPUTexture *, tiles_dilated_fg_tx_, tiles_fg_tx_);
      }
    }
    /* Swap again so that final textures are tiles_dilated_*_tx_. */
    SWAP(GPUFrameBuffer *, tiles_dilate_fb_, tiles_flatten_fb_);
    SWAP(GPUTexture *, tiles_dilated_bg_tx_, tiles_bg_tx_);
    SWAP(GPUTexture *, tiles_dilated_fg_tx_, tiles_fg_tx_);
  }

  /**
   * Create mipmapped color & COC textures for gather passes.
   **/
  void reduce_pass_sync(void)
  {
    DrawEngineType *owner = (DrawEngineType *)view_name_.c_str();
    eGPUSamplerState no_filter = GPU_SAMPLER_DEFAULT;
    /* Divide by 2 because dof_fx_max_coc is in fullres CoC radius and the reduce
     * texture begins at half resolution. */
    float max_space_between_sample = fx_max_coc_ * 0.5f / DOF_GATHER_RING_COUNT;

    int mip_count = max_ii(1, log2_ceil_u(max_space_between_sample));

    reduce_steps_ = mip_count - 1;
    /* This ensure the mipmaps are aligned for the needed 4 mip levels.
     * Starts at 2 because already at half resolution. */
    int multiple = 2 << (mip_count - 1);
    uint res[2] = {(multiple * divide_ceil_u(extent_[0], multiple)) / 2,
                   (multiple * divide_ceil_u(extent_[1], multiple)) / 2};

    uint quater_res[2] = {divide_ceil_u(extent_[0], 4), divide_ceil_u(extent_[1], 4)};

    /* TODO(fclem): Make this dependent of the quality of the gather pass. */
    data_.scatter_coc_threshold = 4.0f;

    {
      DRW_PASS_CREATE(reduce_downsample_ps_, DRW_STATE_WRITE_COLOR);
      GPUShader *sh = shaders_.static_shader_get(DOF_REDUCE_DOWNSAMPLE);
      DRWShadingGroup *grp = DRW_shgroup_create(sh, reduce_downsample_ps_);
      DRW_shgroup_uniform_texture_ref_ex(grp, "color_tx", &setup_color_tx_, no_filter);
      DRW_shgroup_uniform_texture_ref_ex(grp, "coc_tx", &setup_coc_tx_, no_filter);
      DRW_shgroup_call_procedural_triangles(grp, nullptr, 1);

      reduce_downsample_tx_ = DRW_texture_pool_query_2d(UNPACK2(quater_res), GPU_RGBA16F, owner);

      GPU_framebuffer_ensure_config(&reduce_downsample_fb_,
                                    {
                                        GPU_ATTACHMENT_NONE,
                                        GPU_ATTACHMENT_TEXTURE(reduce_downsample_tx_),
                                    });
    }
    {
      DRW_PASS_CREATE(reduce_copy_ps_, DRW_STATE_WRITE_COLOR);
      GPUShader *sh = shaders_.static_shader_get(DOF_REDUCE_COPY);
      DRWShadingGroup *grp = DRW_shgroup_create(sh, reduce_copy_ps_);
      DRW_shgroup_uniform_texture_ref_ex(grp, "color_tx", &setup_color_tx_, no_filter);
      DRW_shgroup_uniform_texture_ref_ex(grp, "coc_tx", &setup_coc_tx_, no_filter);
      DRW_shgroup_uniform_texture_ex(grp, "downsampled_tx", reduce_downsample_tx_, no_filter);
      DRW_shgroup_uniform_block(grp, "dof_block", ubo_);
      DRW_shgroup_call_procedural_triangles(grp, nullptr, 1);

      scatter_src_tx_ = DRW_texture_pool_query_2d(UNPACK2(res), GPU_R11F_G11F_B10F, owner);
    }
    {
      DRW_PASS_CREATE(reduce_recursive_ps_, DRW_STATE_WRITE_COLOR);
      GPUShader *sh = shaders_.static_shader_get(DOF_REDUCE_RECURSIVE);
      DRWShadingGroup *grp = DRW_shgroup_create(sh, reduce_recursive_ps_);
      DRW_shgroup_uniform_texture_ref_ex(grp, "color_tx", &reduced_color_tx_, no_filter);
      DRW_shgroup_uniform_texture_ref_ex(grp, "coc_tx", &reduced_coc_tx_, no_filter);
      DRW_shgroup_call_procedural_triangles(grp, nullptr, 1);
    }

    if (reduced_color_tx_) {
      /* TODO(fclem) In the future, we need to check if mip_count did not change.
       * For now it's ok as we always define all mip level.*/
      if (res[0] != GPU_texture_width(reduced_color_tx_) ||
          res[1] != GPU_texture_width(reduced_color_tx_)) {
        DRW_TEXTURE_FREE_SAFE(reduced_color_tx_);
        DRW_TEXTURE_FREE_SAFE(reduced_coc_tx_);
      }
    }

    if (reduced_color_tx_ == nullptr) {
      /* Color needs to be signed format here. See note in shader for explanation. */
      /* Do not use texture pool because of needs mipmaps. */
      reduced_color_tx_ = GPU_texture_create_2d(
          "dof_reduced_color", UNPACK2(res), mip_count, GPU_RGBA16F, nullptr);
      reduced_coc_tx_ = GPU_texture_create_2d(
          "dof_reduced_coc", UNPACK2(res), mip_count, GPU_R16F, nullptr);
      /* TODO(fclem) Remove once we have immutable storage or when mips are
       * generated on creation. */
      GPU_texture_generate_mipmap(reduced_color_tx_);
      GPU_texture_generate_mipmap(reduced_coc_tx_);
    }

    GPU_framebuffer_ensure_config(&reduce_fb_,
                                  {
                                      GPU_ATTACHMENT_NONE,
                                      GPU_ATTACHMENT_TEXTURE(reduced_color_tx_),
                                      GPU_ATTACHMENT_TEXTURE(reduced_coc_tx_),
                                  });

    GPU_framebuffer_ensure_config(&reduce_copy_fb_,
                                  {
                                      GPU_ATTACHMENT_NONE,
                                      GPU_ATTACHMENT_TEXTURE(reduced_color_tx_),
                                      GPU_ATTACHMENT_TEXTURE(reduced_coc_tx_),
                                      GPU_ATTACHMENT_TEXTURE(scatter_src_tx_),
                                  });
  }

  static void reduce_recusive(void *thunk, int UNUSED(level))
  {
    DepthOfField *dof = reinterpret_cast<DepthOfField *>(thunk);
    DRW_draw_pass(dof->reduce_recursive_ps_);
  }

  void reduce_pass_render(void)
  {
    GPU_framebuffer_bind(reduce_downsample_fb_);
    DRW_draw_pass(reduce_downsample_ps_);

    /* First step is just a copy. */
    GPU_framebuffer_bind(reduce_copy_fb_);
    DRW_draw_pass(reduce_copy_ps_);

    GPU_framebuffer_recursive_downsample(reduce_fb_, reduce_steps_, &reduce_recusive, this);
  }

  /**
   * Do the gather & scatter convolution. For each pixels we gather multiple pixels in its
   * neighborhood depending on the min & max CoC tiles. We apply a median filter on the output.
   * We also scatter a sprite for very bright pixels for high quality bokeh.
   **/
  void convolve_pass_sync(void)
  {
    eGPUSamplerState no_filter = GPU_SAMPLER_DEFAULT;
    eGPUSamplerState with_filter = (GPU_SAMPLER_MIPMAP | GPU_SAMPLER_FILTER);
    DrawEngineType *owner = (DrawEngineType *)view_name_.c_str();
    uint res[2] = {divide_ceil_u(extent_[0], 2), divide_ceil_u(extent_[1], 2)};
    int input_size[2];
    GPU_texture_get_mipmap_size(reduced_color_tx_, 0, input_size);
    for (int i = 0; i < 2; i++) {
      data_.gather_uv_fac[i] = 1.0f / (float)input_size[i];
      data_.texel_size[i] = 1.0f / res[i];
    }

    /* Reuse textures from the setup pass. */
    /* NOTE: We could use the texture pool do that for us but it does not track
     * usage and it might backfire (it does in practice). */
    /* Since it is only used for scatter, and foreground is processed before background, we can
     * reuse the occlusion_tx for both field. */
    occlusion_tx_ = setup_coc_tx_;

    {
      DRW_PASS_CREATE(gather_holefill_ps_, DRW_STATE_WRITE_COLOR);
      GPUShader *sh = shaders_.static_shader_get(DOF_GATHER_HOLEFILL);
      DRWShadingGroup *grp = DRW_shgroup_create(sh, gather_holefill_ps_);
      DRW_shgroup_uniform_texture_ex(grp, "color_bilinear_tx", reduced_color_tx_, with_filter);
      DRW_shgroup_uniform_texture_ex(grp, "color_tx", reduced_color_tx_, no_filter);
      DRW_shgroup_uniform_texture_ex(grp, "coc_tx", reduced_coc_tx_, no_filter);
      DRW_shgroup_uniform_texture(grp, "tiles_bg_tx", tiles_dilated_bg_tx_);
      DRW_shgroup_uniform_texture(grp, "tiles_fg_tx", tiles_dilated_fg_tx_);
      DRW_shgroup_uniform_block(grp, "dof_block", ubo_);
      DRW_shgroup_uniform_block(grp, "sampling_block", sampling_.ubo_get());
      DRW_shgroup_call_procedural_triangles(grp, NULL, 1);

      /* Reuse textures from the setup pass. */
      /* NOTE: We could use the texture pool do that for us but it does not track
       * usage and it might backfire (it does in practice). */
      color_holefill_tx_ = setup_color_tx_;
      weight_holefill_tx_ = DRW_texture_pool_query_2d(UNPACK2(res), GPU_R16F, owner);

      GPU_framebuffer_ensure_config(&gather_holefill_fb_,
                                    {
                                        GPU_ATTACHMENT_NONE,
                                        GPU_ATTACHMENT_TEXTURE(color_holefill_tx_),
                                        GPU_ATTACHMENT_TEXTURE(weight_holefill_tx_),
                                    });
    }
    {
      DRW_PASS_CREATE(gather_fg_ps_, DRW_STATE_WRITE_COLOR);
      GPUShader *sh = shaders_.static_shader_get(bokeh_gather_lut_tx_ ? DOF_GATHER_FOREGROUND_LUT :
                                                                        DOF_GATHER_FOREGROUND);
      DRWShadingGroup *grp = DRW_shgroup_create(sh, gather_fg_ps_);
      DRW_shgroup_uniform_texture_ex(grp, "color_bilinear_tx", reduced_color_tx_, with_filter);
      DRW_shgroup_uniform_texture_ex(grp, "color_tx", reduced_color_tx_, no_filter);
      DRW_shgroup_uniform_texture_ex(grp, "coc_tx", reduced_coc_tx_, no_filter);
      DRW_shgroup_uniform_texture(grp, "tiles_bg_tx", tiles_dilated_bg_tx_);
      DRW_shgroup_uniform_texture(grp, "tiles_fg_tx", tiles_dilated_fg_tx_);
      if (bokeh_gather_lut_tx_) {
        DRW_shgroup_uniform_texture_ref(grp, "bokeh_lut_tx", &bokeh_gather_lut_tx_);
      }
      DRW_shgroup_uniform_block(grp, "dof_block", ubo_);
      DRW_shgroup_uniform_block(grp, "sampling_block", sampling_.ubo_get());
      DRW_shgroup_call_procedural_triangles(grp, NULL, 1);

      color_fg_tx_ = DRW_texture_pool_query_2d(UNPACK2(res), GPU_RGBA16F, owner);
      weight_fg_tx_ = DRW_texture_pool_query_2d(UNPACK2(res), GPU_R16F, owner);
    }
    {
      DRW_PASS_CREATE(gather_bg_ps_, DRW_STATE_WRITE_COLOR);
      GPUShader *sh = shaders_.static_shader_get(bokeh_gather_lut_tx_ ? DOF_GATHER_BACKGROUND_LUT :
                                                                        DOF_GATHER_BACKGROUND);
      DRWShadingGroup *grp = DRW_shgroup_create(sh, gather_bg_ps_);
      DRW_shgroup_uniform_texture_ex(grp, "color_bilinear_tx", reduced_color_tx_, with_filter);
      DRW_shgroup_uniform_texture_ex(grp, "color_tx", reduced_color_tx_, no_filter);
      DRW_shgroup_uniform_texture_ex(grp, "coc_tx", reduced_coc_tx_, no_filter);
      DRW_shgroup_uniform_texture(grp, "tiles_bg_tx", tiles_dilated_bg_tx_);
      DRW_shgroup_uniform_texture(grp, "tiles_fg_tx", tiles_dilated_fg_tx_);
      if (bokeh_gather_lut_tx_) {
        DRW_shgroup_uniform_texture_ref(grp, "bokeh_lut_tx", &bokeh_gather_lut_tx_);
      }
      DRW_shgroup_uniform_block(grp, "dof_block", ubo_);
      DRW_shgroup_uniform_block(grp, "sampling_block", sampling_.ubo_get());
      DRW_shgroup_call_procedural_triangles(grp, NULL, 1);

      color_bg_tx_ = DRW_texture_pool_query_2d(UNPACK2(res), GPU_RGBA16F, owner);
      weight_bg_tx_ = DRW_texture_pool_query_2d(UNPACK2(res), GPU_R16F, owner);
    }

    /* NOTE: First target is holefill texture so we can use the median filter on it and save some
     * texture memory. Both field use the same framebuffer. */
    GPU_framebuffer_ensure_config(&gather_fb_,
                                  {
                                      GPU_ATTACHMENT_NONE,
                                      GPU_ATTACHMENT_TEXTURE(color_holefill_tx_),
                                      GPU_ATTACHMENT_TEXTURE(weight_holefill_tx_),
                                      GPU_ATTACHMENT_TEXTURE(occlusion_tx_),
                                  });
    {
      /**
       * Filter an input buffer using a median filter to reduce noise.
       * NOTE: We use the holefill texture as our input to reduce memory usage.
       * Thus, the holefill pass cannot be filtered.
       **/
      DRW_PASS_CREATE(filter_ps_, DRW_STATE_WRITE_COLOR);
      GPUShader *sh = shaders_.static_shader_get(DOF_FILTER);
      DRWShadingGroup *grp = DRW_shgroup_create(sh, filter_ps_);
      DRW_shgroup_uniform_texture_ex(grp, "color_tx", color_holefill_tx_, no_filter);
      DRW_shgroup_uniform_texture_ex(grp, "weight_tx", weight_holefill_tx_, no_filter);
      DRW_shgroup_call_procedural_triangles(grp, NULL, 1);

      GPU_framebuffer_ensure_config(&filter_fg_fb_,
                                    {
                                        GPU_ATTACHMENT_NONE,
                                        GPU_ATTACHMENT_TEXTURE(color_fg_tx_),
                                        GPU_ATTACHMENT_TEXTURE(weight_fg_tx_),
                                    });

      GPU_framebuffer_ensure_config(&filter_bg_fb_,
                                    {
                                        GPU_ATTACHMENT_NONE,
                                        GPU_ATTACHMENT_TEXTURE(color_bg_tx_),
                                        GPU_ATTACHMENT_TEXTURE(weight_bg_tx_),
                                    });
    }

    /**
     * Do the Scatter convolution. A sprite is emitted for every 4 pixels but is only expanded
     * if the pixels are bright enough to be scattered.
     **/
    data_.scatter_sprite_per_row = input_size[0] / 2;
    int sprite_count = data_.scatter_sprite_per_row * (input_size[1] / 2);
    {
      DRW_PASS_CREATE(scatter_fg_ps_, DRW_STATE_WRITE_COLOR | DRW_STATE_BLEND_ADD_FULL);
      GPUShader *sh = shaders_.static_shader_get(
          bokeh_gather_lut_tx_ ? DOF_SCATTER_FOREGROUND_LUT : DOF_SCATTER_FOREGROUND);
      DRWShadingGroup *grp = DRW_shgroup_create(sh, scatter_fg_ps_);
      DRW_shgroup_uniform_texture_ex(grp, "color_tx", scatter_src_tx_, no_filter);
      DRW_shgroup_uniform_texture_ex(grp, "coc_tx", reduced_coc_tx_, no_filter);
      DRW_shgroup_uniform_texture(grp, "occlusion_tx", occlusion_tx_);
      if (bokeh_scatter_lut_tx_) {
        DRW_shgroup_uniform_texture(grp, "bokehLut", bokeh_scatter_lut_tx_);
      }
      DRW_shgroup_uniform_block(grp, "dof_block", ubo_);
      DRW_shgroup_call_procedural_triangles(grp, NULL, sprite_count);

      GPU_framebuffer_ensure_config(&scatter_fg_fb_,
                                    {
                                        GPU_ATTACHMENT_NONE,
                                        GPU_ATTACHMENT_TEXTURE(color_fg_tx_),
                                    });
    }
    {
      DRW_PASS_CREATE(scatter_bg_ps_, DRW_STATE_WRITE_COLOR | DRW_STATE_BLEND_ADD_FULL);
      GPUShader *sh = shaders_.static_shader_get(
          bokeh_gather_lut_tx_ ? DOF_SCATTER_BACKGROUND_LUT : DOF_SCATTER_BACKGROUND);
      DRWShadingGroup *grp = DRW_shgroup_create(sh, scatter_bg_ps_);
      DRW_shgroup_uniform_texture_ex(grp, "color_tx", scatter_src_tx_, no_filter);
      DRW_shgroup_uniform_texture_ex(grp, "coc_tx", reduced_coc_tx_, no_filter);
      DRW_shgroup_uniform_texture(grp, "occlusion_tx", occlusion_tx_);
      if (bokeh_scatter_lut_tx_) {
        DRW_shgroup_uniform_texture(grp, "bokehLut", bokeh_scatter_lut_tx_);
      }
      DRW_shgroup_uniform_block(grp, "dof_block", ubo_);
      DRW_shgroup_call_procedural_triangles(grp, NULL, sprite_count);

      GPU_framebuffer_ensure_config(&scatter_bg_fb_,
                                    {
                                        GPU_ATTACHMENT_NONE,
                                        GPU_ATTACHMENT_TEXTURE(color_bg_tx_),
                                    });
    }
  }

  void convolve_pass_render(void)
  {
    DRW_stats_group_start("Foreground convolution");
    GPU_framebuffer_bind(gather_fb_);
    DRW_draw_pass(gather_fg_ps_);

    GPU_framebuffer_bind(filter_fg_fb_);
    DRW_draw_pass(filter_ps_);

    GPU_framebuffer_bind(scatter_fg_fb_);
    DRW_draw_pass(scatter_fg_ps_);
    DRW_stats_group_end();

    DRW_stats_group_start("Background convolution");
    GPU_framebuffer_bind(gather_fb_);
    DRW_draw_pass(gather_bg_ps_);

    GPU_framebuffer_bind(filter_bg_fb_);
    DRW_draw_pass(filter_ps_);

    GPU_framebuffer_bind(scatter_bg_fb_);
    DRW_draw_pass(scatter_bg_ps_);
    DRW_stats_group_end();

    DRW_stats_group_start("Background convolution");
    /* Hole-fill convolution. */
    GPU_framebuffer_bind(gather_holefill_fb_);
    DRW_draw_pass(gather_holefill_ps_);
    /* NOTE: We do not filter the hole-fill pass as we use it as out filter input
     * buffer. Also effect is likely to not be noticeable. */
    DRW_stats_group_end();
  }

  /**
   * Recombine the result of the foreground and background processing.
   * Also perform a slight out of focus gather to improve geometric continuity.
   **/
  void resolve_pass_sync(void)
  {
    eGPUSamplerState no_filter = GPU_SAMPLER_DEFAULT;
    eGPUSamplerState with_filter = GPU_SAMPLER_FILTER;
    eShaderType sh_type = (bokeh_resolve_lut_tx_) ?
                              (do_hq_slight_focus_ ? DOF_RESOLVE_LUT_HQ : DOF_RESOLVE_LUT) :
                              (do_hq_slight_focus_ ? DOF_RESOLVE_HQ : DOF_RESOLVE);

    DRW_PASS_CREATE(resolve_ps_, DRW_STATE_WRITE_COLOR);
    GPUShader *sh = shaders_.static_shader_get(sh_type);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, resolve_ps_);
    DRW_shgroup_uniform_texture_ref_ex(grp, "depth_tx", &input_depth_tx_, no_filter);
    DRW_shgroup_uniform_texture_ref_ex(grp, "color_tx", &input_color_tx_, no_filter);
    DRW_shgroup_uniform_texture_ex(grp, "color_bg_tx", color_bg_tx_, with_filter);
    DRW_shgroup_uniform_texture_ex(grp, "color_fg_tx", color_fg_tx_, with_filter);
    DRW_shgroup_uniform_texture_ex(grp, "color_holefill_tx", color_holefill_tx_, with_filter);
    DRW_shgroup_uniform_texture(grp, "tiles_bg_tx", tiles_dilated_bg_tx_);
    DRW_shgroup_uniform_texture(grp, "tiles_fg_tx", tiles_dilated_fg_tx_);
    DRW_shgroup_uniform_texture_ex(grp, "weight_bg_tx", weight_bg_tx_, with_filter);
    DRW_shgroup_uniform_texture_ex(grp, "weight_fg_tx", weight_fg_tx_, with_filter);
    DRW_shgroup_uniform_texture_ex(grp, "weight_holefill_tx", weight_holefill_tx_, with_filter);
    DRW_shgroup_uniform_block(grp, "dof_block", ubo_);
    DRW_shgroup_uniform_block(grp, "sampling_block", sampling_.ubo_get());
    if (bokeh_resolve_lut_tx_) {
      DRW_shgroup_uniform_texture_ref(grp, "bokeh_lut_tx", &bokeh_resolve_lut_tx_);
    }
    DRW_shgroup_call_procedural_triangles(grp, NULL, 1);
  }

  void resolve_pass_render(GPUTexture *output_tx)
  {
    GPU_framebuffer_ensure_config(&resolve_fb_,
                                  {
                                      GPU_ATTACHMENT_NONE,
                                      GPU_ATTACHMENT_TEXTURE(output_tx),
                                  });

    GPU_framebuffer_bind(resolve_fb_);
    DRW_draw_pass(resolve_ps_);
  }
};

/** \} */

}  // namespace blender::eevee