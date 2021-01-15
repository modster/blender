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
 * Copyright 2016, Blender Foundation.
 */

/** \file
 * \ingroup draw_engine
 *
 * Depth of field post process effect.
 */

#include "DRW_render.h"

#include "DNA_camera_types.h"
#include "DNA_screen_types.h"
#include "DNA_view3d_types.h"
#include "DNA_world_types.h"

#include "BKE_camera.h"

#include "BLI_string_utils.h"

#include "DEG_depsgraph.h"
#include "DEG_depsgraph_query.h"

#include "GPU_framebuffer.h"
#include "GPU_texture.h"
#include "eevee_private.h"

int EEVEE_depth_of_field_init(EEVEE_ViewLayerData *UNUSED(sldata),
                              EEVEE_Data *vedata,
                              Object *camera)
{
  EEVEE_TextureList *txl = vedata->txl;
  EEVEE_StorageList *stl = vedata->stl;
  EEVEE_FramebufferList *fbl = vedata->fbl;
  EEVEE_EffectsInfo *effects = stl->effects;

  const DRWContextState *draw_ctx = DRW_context_state_get();
  const Scene *scene_eval = DEG_get_evaluated_scene(draw_ctx->depsgraph);

  Camera *cam = (camera != NULL) ? camera->data : NULL;

  if (cam && (cam->dof.flag & CAM_DOF_ENABLED)) {
    RegionView3D *rv3d = draw_ctx->rv3d;
    const float *viewport_size = DRW_viewport_size_get();

    /* Retrieve Near and Far distance */
    effects->dof_coc_near_dist = -cam->clip_start;
    effects->dof_coc_far_dist = -cam->clip_end;

    /* Parameters */
    float fstop = cam->dof.aperture_fstop;
    float blades = cam->dof.aperture_blades;
    float rotation = cam->dof.aperture_rotation;
    float ratio = 1.0f / cam->dof.aperture_ratio;
    float sensor = BKE_camera_sensor_size(cam->sensor_fit, cam->sensor_x, cam->sensor_y);
    float focus_dist = BKE_camera_object_dof_distance(camera);
    float focal_len = cam->lens;

    const float scale_camera = 0.001f;
    /* we want radius here for the aperture number  */
    float aperture = 0.5f * scale_camera * focal_len / fstop;
    float focal_len_scaled = scale_camera * focal_len;
    float sensor_scaled = scale_camera * sensor;

    if (rv3d != NULL) {
      sensor_scaled *= rv3d->viewcamtexcofac[0];
    }

    effects->dof_coc_params[1] = -aperture *
                                 fabsf(focal_len_scaled / (focus_dist - focal_len_scaled));
    effects->dof_coc_params[1] *= viewport_size[0] / sensor_scaled;
    effects->dof_coc_params[0] = -focus_dist * effects->dof_coc_params[1];

    effects->dof_bokeh[0] = rotation;
    effects->dof_bokeh[1] = ratio;
    effects->dof_bokeh[2] = scene_eval->eevee.bokeh_max_size;

    /* TODO(fclem) User parameters. */
    effects->dof_scatter_color_threshold = 1.5f;

    /* Precompute values to save instructions in fragment shader. */
    effects->dof_bokeh_sides[0] = blades;
    effects->dof_bokeh_sides[1] = blades > 0.0f ? 2.0f * M_PI / blades : 0.0f;
    effects->dof_bokeh_sides[2] = blades / (2.0f * M_PI);
    effects->dof_bokeh_sides[3] = blades > 0.0f ? cosf(M_PI / blades) : 0.0f;

    return EFFECT_DOF | EFFECT_POST_BUFFER;
  }

  /* Cleanup to release memory */
  GPU_FRAMEBUFFER_FREE_SAFE(fbl->dof_setup_fb);
  GPU_FRAMEBUFFER_FREE_SAFE(fbl->dof_flatten_tiles_fb);
  GPU_FRAMEBUFFER_FREE_SAFE(fbl->dof_dilate_tiles_fb);
  GPU_FRAMEBUFFER_FREE_SAFE(fbl->dof_reduce_fb);
  GPU_FRAMEBUFFER_FREE_SAFE(fbl->dof_gather_fg_fb);
  GPU_FRAMEBUFFER_FREE_SAFE(fbl->dof_gather_bg_fb);
  GPU_FRAMEBUFFER_FREE_SAFE(fbl->dof_scatter_bg_fb);
  DRW_TEXTURE_FREE_SAFE(txl->dof_reduced_color);
  DRW_TEXTURE_FREE_SAFE(txl->dof_reduced_coc);

  return 0;
}

#define NO_FILTERING GPU_SAMPLER_MIPMAP
#define COLOR_FORMAT GPU_RGBA16F
#define FG_TILE_FORMAT GPU_R11F_G11F_B10F
#define BG_TILE_FORMAT GPU_R11F_G11F_B10F
#define TILE_DIVISOR 16

/**
 * Ouputs halfResColorBuffer and halfResCocBuffer.
 **/
static void dof_setup_pass_init(EEVEE_FramebufferList *fbl,
                                EEVEE_PassList *psl,
                                EEVEE_EffectsInfo *fx)
{
  DefaultTextureList *dtxl = DRW_viewport_texture_list_get();

  void *owner = (void *)&EEVEE_depth_of_field_init;
  const float *fullres = DRW_viewport_size_get();
  int res[2] = {divide_ceil_u(fullres[0], 2), divide_ceil_u(fullres[1], 2)};

  DRW_PASS_CREATE(psl->dof_setup, DRW_STATE_WRITE_COLOR);

  GPUShader *sh = EEVEE_shaders_depth_of_field_setup_get();
  DRWShadingGroup *grp = DRW_shgroup_create(sh, psl->dof_setup);
  DRW_shgroup_uniform_texture_ref_ex(grp, "colorBuffer", &fx->source_buffer, NO_FILTERING);
  DRW_shgroup_uniform_texture_ref_ex(grp, "depthBuffer", &dtxl->depth, NO_FILTERING);
  DRW_shgroup_uniform_vec4_copy(grp, "cocParams", fx->dof_coc_params);
  DRW_shgroup_uniform_vec4_copy(grp, "bokehParams", fx->dof_bokeh);
  DRW_shgroup_call(grp, DRW_cache_fullscreen_quad_get(), NULL);

  fx->dof_half_res_color_tx = DRW_texture_pool_query_2d(UNPACK2(res), COLOR_FORMAT, owner);
  fx->dof_half_res_coc_tx = DRW_texture_pool_query_2d(UNPACK2(res), GPU_RG16F, owner);

  GPU_framebuffer_ensure_config(&fbl->dof_setup_fb,
                                {
                                    GPU_ATTACHMENT_NONE,
                                    GPU_ATTACHMENT_TEXTURE(fx->dof_half_res_color_tx),
                                    GPU_ATTACHMENT_TEXTURE(fx->dof_half_res_coc_tx),
                                });
}

/**
 * Ouputs min & max coc in each 8x8 half res pixel tiles (so 1/16th of fullres).
 **/
static void dof_flatten_tiles_pass_init(EEVEE_FramebufferList *fbl,
                                        EEVEE_PassList *psl,
                                        EEVEE_EffectsInfo *fx)
{
  void *owner = (void *)&EEVEE_depth_of_field_init;
  const float *fullres = DRW_viewport_size_get();
  int res[2] = {divide_ceil_u(fullres[0], TILE_DIVISOR), divide_ceil_u(fullres[1], TILE_DIVISOR)};

  DRW_PASS_CREATE(psl->dof_flatten_tiles, DRW_STATE_WRITE_COLOR);

  GPUShader *sh = EEVEE_shaders_depth_of_field_flatten_tiles_get();
  DRWShadingGroup *grp = DRW_shgroup_create(sh, psl->dof_flatten_tiles);
  DRW_shgroup_uniform_texture_ref_ex(
      grp, "halfResCocBuffer", &fx->dof_half_res_coc_tx, NO_FILTERING);
  DRW_shgroup_call(grp, DRW_cache_fullscreen_quad_get(), NULL);

  fx->dof_coc_tiles_fg_tx = DRW_texture_pool_query_2d(UNPACK2(res), FG_TILE_FORMAT, owner);
  fx->dof_coc_tiles_bg_tx = DRW_texture_pool_query_2d(UNPACK2(res), BG_TILE_FORMAT, owner);

  GPU_framebuffer_ensure_config(&fbl->dof_flatten_tiles_fb,
                                {
                                    GPU_ATTACHMENT_NONE,
                                    GPU_ATTACHMENT_TEXTURE(fx->dof_coc_tiles_fg_tx),
                                    GPU_ATTACHMENT_TEXTURE(fx->dof_coc_tiles_bg_tx),
                                });
}

/**
 * Dilates the min & max cocs to cover maximum coc values.
 * Output format/dimensions should be the same as coc_flatten_pass as they are swapped for
 * doing multiple dilation passes.
 **/
static void dof_dilate_tiles_pass_init(EEVEE_FramebufferList *fbl,
                                       EEVEE_PassList *psl,
                                       EEVEE_EffectsInfo *fx)
{
  void *owner = (void *)&EEVEE_depth_of_field_init;
  const float *fullres = DRW_viewport_size_get();
  int res[2] = {divide_ceil_u(fullres[0], TILE_DIVISOR), divide_ceil_u(fullres[1], TILE_DIVISOR)};

  fx->dof_dilate_steps = 1; /* TODO */

  DRW_PASS_CREATE(psl->dof_dilate_tiles_minmax, DRW_STATE_WRITE_COLOR);
  DRW_PASS_CREATE(psl->dof_dilate_tiles_minabs, DRW_STATE_WRITE_COLOR);

  for (int pass = 0; pass < 2; pass++) {
    DRWPass *drw_pass = (pass == 0) ? psl->dof_dilate_tiles_minmax : psl->dof_dilate_tiles_minabs;
    GPUShader *sh = EEVEE_shaders_depth_of_field_dilate_tiles_get(pass);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, drw_pass);
    DRW_shgroup_uniform_texture_ref(grp, "cocTilesFgBuffer", &fx->dof_coc_tiles_fg_tx);
    DRW_shgroup_uniform_texture_ref(grp, "cocTilesBgBuffer", &fx->dof_coc_tiles_bg_tx);
    DRW_shgroup_call(grp, DRW_cache_fullscreen_quad_get(), NULL);
  }

  fx->dof_coc_dilated_tiles_fg_tx = DRW_texture_pool_query_2d(UNPACK2(res), FG_TILE_FORMAT, owner);
  fx->dof_coc_dilated_tiles_bg_tx = DRW_texture_pool_query_2d(UNPACK2(res), BG_TILE_FORMAT, owner);

  GPU_framebuffer_ensure_config(&fbl->dof_dilate_tiles_fb,
                                {
                                    GPU_ATTACHMENT_NONE,
                                    GPU_ATTACHMENT_TEXTURE(fx->dof_coc_dilated_tiles_fg_tx),
                                    GPU_ATTACHMENT_TEXTURE(fx->dof_coc_dilated_tiles_bg_tx),
                                });
}

/**
 * Create mipmaped color & coc textures for gather passes.
 **/
static void dof_reduce_pass_init(EEVEE_FramebufferList *fbl,
                                 EEVEE_PassList *psl,
                                 EEVEE_TextureList *txl,
                                 EEVEE_EffectsInfo *fx)
{
  const float *fullres = DRW_viewport_size_get();
  int mip_count = 4; /* TODO(fclem) depends on max blur radius. */
  fx->dof_reduce_steps = mip_count - 1;
  /* This ensure the mipmaps are aligned for the needed 4 mip levels.
   * Starts at 2 because already at half resolution. */
  int multiple = 2 << (mip_count - 1);
  int res[2] = {(multiple * divide_ceil_u(fullres[0], multiple)) / 2,
                (multiple * divide_ceil_u(fullres[1], multiple)) / 2};

  /* TODO(fclem): Make this dependent of the quality of the gather pass. */
  fx->dof_scatter_coc_threshold = 5.0f;

  {
    DRW_PASS_CREATE(psl->dof_reduce_copy, DRW_STATE_WRITE_COLOR);

    const bool is_copy_pass = true;
    GPUShader *sh = EEVEE_shaders_depth_of_field_reduce_get(is_copy_pass);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, psl->dof_reduce_copy);
    DRW_shgroup_uniform_texture_ref_ex(
        grp, "colorBuffer", &fx->dof_reduce_input_color_tx, NO_FILTERING);
    DRW_shgroup_uniform_texture_ref_ex(
        grp, "cocBuffer", &fx->dof_reduce_input_coc_tx, NO_FILTERING);
    DRW_shgroup_uniform_float_copy(grp, "scatterColorThreshold", fx->dof_scatter_color_threshold);
    DRW_shgroup_uniform_float_copy(grp, "scatterCocThreshold", fx->dof_scatter_coc_threshold);
    DRW_shgroup_call(grp, DRW_cache_fullscreen_quad_get(), NULL);
  }

  {
    DRW_PASS_CREATE(psl->dof_reduce, DRW_STATE_WRITE_COLOR);

    const bool is_copy_pass = false;
    GPUShader *sh = EEVEE_shaders_depth_of_field_reduce_get(is_copy_pass);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, psl->dof_reduce);
    DRW_shgroup_uniform_texture_ref_ex(
        grp, "colorBuffer", &fx->dof_reduce_input_color_tx, NO_FILTERING);
    DRW_shgroup_uniform_texture_ref_ex(
        grp, "cocBuffer", &fx->dof_reduce_input_coc_tx, NO_FILTERING);
    DRW_shgroup_uniform_float_copy(grp, "scatterColorThreshold", fx->dof_scatter_color_threshold);
    DRW_shgroup_call(grp, DRW_cache_fullscreen_quad_get(), NULL);
  }

  if (txl->dof_reduced_color == NULL) {
    /* Color needs to be signed format here. See note in shader for explanation. */
    /* Do not use texture pool because of needs mipmaps. */
    txl->dof_reduced_color = GPU_texture_create_2d(
        "dof_reduced_color", UNPACK2(res), mip_count, GPU_RGBA16F, NULL);
    txl->dof_reduced_coc = GPU_texture_create_2d(
        "dof_reduced_coc", UNPACK2(res), mip_count, GPU_R16F, NULL);

    /* TODO(fclem) Remove once we have immutable storage or when mips are generated on creation. */
    GPU_texture_generate_mipmap(txl->dof_reduced_color);
    GPU_texture_generate_mipmap(txl->dof_reduced_coc);
  }

  GPU_framebuffer_ensure_config(&fbl->dof_reduce_fb,
                                {
                                    GPU_ATTACHMENT_NONE,
                                    GPU_ATTACHMENT_TEXTURE(txl->dof_reduced_color),
                                    GPU_ATTACHMENT_TEXTURE(txl->dof_reduced_coc),
                                });
}

/**
 * Do the gather convolution. For each pixels we gather multiple pixels in its neighborhood
 * depending on the min & max CoC tiles.
 **/
static void dof_gather_pass_init(EEVEE_FramebufferList *fbl,
                                 EEVEE_PassList *psl,
                                 EEVEE_TextureList *txl,
                                 EEVEE_EffectsInfo *fx)
{
  void *owner = (void *)&dof_gather_pass_init;
  const float *fullres = DRW_viewport_size_get();
  int res[2] = {divide_ceil_u(fullres[0], 2), divide_ceil_u(fullres[1], 2)};
  int input_size[2];
  GPU_texture_get_mipmap_size(txl->dof_reduced_color, 0, input_size);
  float uv_correction_fac[2] = {res[0] / (float)input_size[0], res[1] / (float)input_size[1]};
  float output_texel_size[2] = {1.0f / res[0], 1.0f / res[1]};

  {
    DRW_PASS_CREATE(psl->dof_gather_fg, DRW_STATE_WRITE_COLOR);

    const bool is_foreground = true;
    GPUShader *sh = EEVEE_shaders_depth_of_field_gather_get(is_foreground);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, psl->dof_gather_fg);
    DRW_shgroup_uniform_texture_ref_ex(grp, "colorBuffer", &txl->dof_reduced_color, NO_FILTERING);
    DRW_shgroup_uniform_texture_ref_ex(grp, "cocBuffer", &txl->dof_reduced_coc, NO_FILTERING);
    DRW_shgroup_uniform_texture_ref(grp, "cocTilesBuffer", &fx->dof_coc_dilated_tiles_fg_tx);
    DRW_shgroup_uniform_texture(grp, "utilTex", EEVEE_materials_get_util_tex());
    DRW_shgroup_uniform_float_copy(grp, "scatterColorThreshold", fx->dof_scatter_color_threshold);
    DRW_shgroup_uniform_vec2_copy(grp, "gatherInputUvCorrection", uv_correction_fac);
    DRW_shgroup_uniform_vec2_copy(grp, "gatherOutputTexelSize", output_texel_size);
    DRW_shgroup_uniform_vec4(grp, "bokehParams", fx->dof_bokeh, 2);
    DRW_shgroup_call(grp, DRW_cache_fullscreen_quad_get(), NULL);

    /* NOTE: This pass is the owner. So textures from pool can come from previous passes. */
    fx->dof_fg_color_tx = DRW_texture_pool_query_2d(UNPACK2(res), COLOR_FORMAT, owner);
    fx->dof_fg_weight_tx = DRW_texture_pool_query_2d(UNPACK2(res), GPU_R16F, owner);

    GPU_framebuffer_ensure_config(&fbl->dof_gather_fg_fb,
                                  {
                                      GPU_ATTACHMENT_NONE,
                                      GPU_ATTACHMENT_TEXTURE(fx->dof_fg_color_tx),
                                      GPU_ATTACHMENT_TEXTURE(fx->dof_fg_weight_tx),
                                  });
  }
  {
    DRW_PASS_CREATE(psl->dof_gather_bg, DRW_STATE_WRITE_COLOR);

    const bool is_foreground = false;
    GPUShader *sh = EEVEE_shaders_depth_of_field_gather_get(is_foreground);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, psl->dof_gather_bg);
    DRW_shgroup_uniform_texture_ref_ex(grp, "colorBuffer", &txl->dof_reduced_color, NO_FILTERING);
    DRW_shgroup_uniform_texture_ref_ex(grp, "cocBuffer", &txl->dof_reduced_coc, NO_FILTERING);
    DRW_shgroup_uniform_texture_ref(grp, "cocTilesBuffer", &fx->dof_coc_dilated_tiles_bg_tx);
    DRW_shgroup_uniform_texture(grp, "utilTex", EEVEE_materials_get_util_tex());
    DRW_shgroup_uniform_float_copy(grp, "scatterColorThreshold", fx->dof_scatter_color_threshold);
    DRW_shgroup_uniform_vec2_copy(grp, "gatherInputUvCorrection", uv_correction_fac);
    DRW_shgroup_uniform_vec2_copy(grp, "gatherOutputTexelSize", output_texel_size);
    DRW_shgroup_uniform_vec4(grp, "bokehParams", fx->dof_bokeh, 2);
    DRW_shgroup_call(grp, DRW_cache_fullscreen_quad_get(), NULL);

    /* NOTE: This pass is the owner. So textures from pool can come from previous passes. */
    fx->dof_bg_color_tx = DRW_texture_pool_query_2d(UNPACK2(res), COLOR_FORMAT, owner);
    fx->dof_bg_weight_tx = DRW_texture_pool_query_2d(UNPACK2(res), GPU_R16F, owner);
    /* TODO(fclem): scatter occlusion. */
    // fx->dof_bg_occlusion_tx = DRW_texture_pool_query_2d(UNPACK2(res), GPU_RG16F, owner);

    GPU_framebuffer_ensure_config(&fbl->dof_gather_bg_fb,
                                  {
                                      GPU_ATTACHMENT_NONE,
                                      GPU_ATTACHMENT_TEXTURE(fx->dof_bg_color_tx),
                                      GPU_ATTACHMENT_TEXTURE(fx->dof_bg_weight_tx),
                                  });
  }
}

/**
 * Do the Scatter convolution. A sprite is emited for every 4 pixels but is only expanded if the
 * pixels are bright enough to be scattered.
 **/
static void dof_scatter_pass_init(EEVEE_FramebufferList *fbl,
                                  EEVEE_PassList *psl,
                                  EEVEE_TextureList *txl,
                                  EEVEE_EffectsInfo *fx)
{
  int input_size[2], target_size[2];
  GPU_texture_get_mipmap_size(fx->dof_half_res_color_tx, 0, input_size);
  GPU_texture_get_mipmap_size(fx->dof_bg_color_tx, 0, target_size);
  /* Draw a sprite for every four halfres pixels. */
  int sprite_count = (input_size[0] / 2) * (input_size[1] / 2);
  float target_texel_size[2] = {1.0f / target_size[0], 1.0f / target_size[1]};

  {
    DRW_PASS_CREATE(psl->dof_scatter_fg, DRW_STATE_WRITE_COLOR | DRW_STATE_BLEND_ADD_FULL);

    const bool is_foreground = true;
    GPUShader *sh = EEVEE_shaders_depth_of_field_scatter_get(is_foreground);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, psl->dof_scatter_fg);
    DRW_shgroup_uniform_texture_ref_ex(grp, "colorBuffer", &txl->dof_reduced_color, NO_FILTERING);
    DRW_shgroup_uniform_texture_ref_ex(grp, "cocBuffer", &txl->dof_reduced_coc, NO_FILTERING);
    DRW_shgroup_uniform_float_copy(grp, "scatterColorThreshold", fx->dof_scatter_color_threshold);
    DRW_shgroup_uniform_float_copy(grp, "scatterCocThreshold", fx->dof_scatter_coc_threshold);
    DRW_shgroup_uniform_vec2_copy(grp, "targetTexelSize", target_texel_size);
    DRW_shgroup_uniform_int_copy(grp, "spritePerRow", input_size[0] / 2);
    DRW_shgroup_uniform_vec4(grp, "bokehParams", fx->dof_bokeh, 2);
    DRW_shgroup_call_procedural_triangles(grp, NULL, sprite_count);

    GPU_framebuffer_ensure_config(&fbl->dof_scatter_fg_fb,
                                  {
                                      GPU_ATTACHMENT_NONE,
                                      GPU_ATTACHMENT_TEXTURE(fx->dof_fg_color_tx),
                                  });
  }
  {
    DRW_PASS_CREATE(psl->dof_scatter_bg, DRW_STATE_WRITE_COLOR | DRW_STATE_BLEND_ADD_FULL);

    const bool is_foreground = false;
    GPUShader *sh = EEVEE_shaders_depth_of_field_scatter_get(is_foreground);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, psl->dof_scatter_bg);
    DRW_shgroup_uniform_texture_ref_ex(grp, "colorBuffer", &txl->dof_reduced_color, NO_FILTERING);
    DRW_shgroup_uniform_texture_ref_ex(grp, "cocBuffer", &txl->dof_reduced_coc, NO_FILTERING);
    DRW_shgroup_uniform_float_copy(grp, "scatterColorThreshold", fx->dof_scatter_color_threshold);
    DRW_shgroup_uniform_float_copy(grp, "scatterCocThreshold", fx->dof_scatter_coc_threshold);
    DRW_shgroup_uniform_vec2_copy(grp, "targetTexelSize", target_texel_size);
    DRW_shgroup_uniform_int_copy(grp, "spritePerRow", input_size[0] / 2);
    DRW_shgroup_uniform_vec4(grp, "bokehParams", fx->dof_bokeh, 2);
    /* TODO(fclem): scatter occlusion. */
    // DRW_shgroup_uniform_texture_ref(grp, "occlusionBuffer", &fx->dof_bg_occlusion_tx);
    DRW_shgroup_call_procedural_triangles(grp, NULL, sprite_count);

    GPU_framebuffer_ensure_config(&fbl->dof_scatter_bg_fb,
                                  {
                                      GPU_ATTACHMENT_NONE,
                                      GPU_ATTACHMENT_TEXTURE(fx->dof_bg_color_tx),
                                  });
  }
}

/**
 * Recombine the result of the foreground and background processing. Also perform a slight out of
 * focus blur to improve geometric continuity.
 **/
static void dof_recombine_pass_init(EEVEE_FramebufferList *UNUSED(fbl),
                                    EEVEE_PassList *psl,
                                    EEVEE_EffectsInfo *fx)
{
  DefaultTextureList *dtxl = DRW_viewport_texture_list_get();

  DRW_PASS_CREATE(psl->dof_resolve, DRW_STATE_WRITE_COLOR);

  GPUShader *sh = EEVEE_shaders_depth_of_field_resolve_get();
  DRWShadingGroup *grp = DRW_shgroup_create(sh, psl->dof_resolve);
  DRW_shgroup_uniform_texture_ref(grp, "fullResColorBuffer", &fx->source_buffer);
  DRW_shgroup_uniform_texture_ref(grp, "fullResDepthBuffer", &dtxl->depth);
  DRW_shgroup_uniform_texture_ref_ex(grp, "bgColorBuffer", &fx->dof_bg_color_tx, NO_FILTERING);
  DRW_shgroup_uniform_texture_ref_ex(grp, "bgWeightBuffer", &fx->dof_bg_weight_tx, NO_FILTERING);
  DRW_shgroup_uniform_texture_ref(grp, "bgTileBuffer", &fx->dof_coc_dilated_tiles_bg_tx);
  DRW_shgroup_uniform_texture_ref_ex(grp, "fgColorBuffer", &fx->dof_fg_color_tx, NO_FILTERING);
  DRW_shgroup_uniform_texture_ref_ex(grp, "fgWeightBuffer", &fx->dof_fg_weight_tx, NO_FILTERING);
  DRW_shgroup_uniform_texture_ref(grp, "fgTileBuffer", &fx->dof_coc_dilated_tiles_fg_tx);
  DRW_shgroup_uniform_texture(grp, "utilTex", EEVEE_materials_get_util_tex());
  DRW_shgroup_uniform_vec4_copy(grp, "cocParams", fx->dof_coc_params);
  DRW_shgroup_call(grp, DRW_cache_fullscreen_quad_get(), NULL);
}

void EEVEE_depth_of_field_cache_init(EEVEE_ViewLayerData *UNUSED(sldata), EEVEE_Data *vedata)
{
  EEVEE_TextureList *txl = vedata->txl;
  EEVEE_FramebufferList *fbl = vedata->fbl;
  EEVEE_PassList *psl = vedata->psl;
  EEVEE_StorageList *stl = vedata->stl;
  EEVEE_EffectsInfo *fx = stl->effects;

  if ((fx->enabled_effects & EFFECT_DOF) != 0) {
    dof_setup_pass_init(fbl, psl, fx);
    dof_flatten_tiles_pass_init(fbl, psl, fx);
    dof_dilate_tiles_pass_init(fbl, psl, fx);
    dof_reduce_pass_init(fbl, psl, txl, fx);
    dof_gather_pass_init(fbl, psl, txl, fx);
    // dof_filter_pass_init(fbl, psl, fx); /* TODO(fclem) */
    dof_scatter_pass_init(fbl, psl, txl, fx);
    dof_recombine_pass_init(fbl, psl, fx);
  }
}

static void dof_recursive_reduce(void *vedata, int UNUSED(level))
{
  EEVEE_PassList *psl = ((EEVEE_Data *)vedata)->psl;
  EEVEE_TextureList *txl = ((EEVEE_Data *)vedata)->txl;
  EEVEE_EffectsInfo *fx = ((EEVEE_Data *)vedata)->stl->effects;

  fx->dof_reduce_input_color_tx = txl->dof_reduced_color;
  fx->dof_reduce_input_coc_tx = txl->dof_reduced_coc;

  DRW_draw_pass(psl->dof_reduce);
}

void EEVEE_depth_of_field_draw(EEVEE_Data *vedata)
{
  EEVEE_PassList *psl = vedata->psl;
  EEVEE_TextureList *txl = vedata->txl;
  EEVEE_FramebufferList *fbl = vedata->fbl;
  EEVEE_StorageList *stl = vedata->stl;
  EEVEE_EffectsInfo *effects = stl->effects; /* TODO(fclem): Because of silly SWAP_BUFFERS. */
  EEVEE_EffectsInfo *fx = effects;

  /* Depth Of Field */
  if ((effects->enabled_effects & EFFECT_DOF) != 0) {
    DRW_stats_group_start("Depth of Field");

    GPU_framebuffer_bind(fbl->dof_setup_fb);
    DRW_draw_pass(psl->dof_setup);

    GPU_framebuffer_bind(fbl->dof_flatten_tiles_fb);
    DRW_draw_pass(psl->dof_flatten_tiles);

    for (int pass = 0; pass < 2; pass++) {
      DRWPass *drw_pass = (pass == 0) ? psl->dof_dilate_tiles_minmax :
                                        psl->dof_dilate_tiles_minabs;

      for (int i = 0; i < fx->dof_dilate_steps; i++) {
        GPU_framebuffer_bind(fbl->dof_dilate_tiles_fb);
        DRW_draw_pass(drw_pass);

        SWAP(GPUFrameBuffer *, fbl->dof_dilate_tiles_fb, fbl->dof_flatten_tiles_fb);
        SWAP(GPUTexture *, fx->dof_coc_dilated_tiles_bg_tx, fx->dof_coc_tiles_bg_tx);
        SWAP(GPUTexture *, fx->dof_coc_dilated_tiles_fg_tx, fx->dof_coc_tiles_fg_tx);
      }
    }
    /* Swap again so that final textures are dof_coc_dilated_tiles_*_tx. */
    SWAP(GPUFrameBuffer *, fbl->dof_dilate_tiles_fb, fbl->dof_flatten_tiles_fb);
    SWAP(GPUTexture *, fx->dof_coc_dilated_tiles_bg_tx, fx->dof_coc_tiles_bg_tx);
    SWAP(GPUTexture *, fx->dof_coc_dilated_tiles_fg_tx, fx->dof_coc_tiles_fg_tx);

    /* First step is just a copy. */
    fx->dof_reduce_input_color_tx = fx->dof_half_res_color_tx;
    fx->dof_reduce_input_coc_tx = fx->dof_half_res_coc_tx;
    GPU_framebuffer_bind(fbl->dof_reduce_fb);
    DRW_draw_pass(psl->dof_reduce_copy);

    GPU_framebuffer_recursive_downsample(
        fbl->dof_reduce_fb, fx->dof_reduce_steps, &dof_recursive_reduce, vedata);

    {
      /* Foreground convolution. */
      GPU_framebuffer_bind(fbl->dof_gather_fg_fb);
      DRW_draw_pass(psl->dof_gather_fg);

      // DRW_draw_pass(psl->dof_filter_fg); /* TODO(fclem): gather filter. */

      GPU_framebuffer_bind(fbl->dof_scatter_fg_fb);
      DRW_draw_pass(psl->dof_scatter_fg);
    }

    {
      /* Background convolution. */
      GPU_framebuffer_bind(fbl->dof_gather_bg_fb);
      DRW_draw_pass(psl->dof_gather_bg);

      // DRW_draw_pass(psl->dof_filter_bg); /* TODO(fclem): gather filter. */

      GPU_framebuffer_bind(fbl->dof_scatter_bg_fb);
      DRW_draw_pass(psl->dof_scatter_bg);
    }

    GPU_framebuffer_bind(fx->target_buffer);
    DRW_draw_pass(psl->dof_resolve);

    SWAP_BUFFERS();

    DRW_stats_group_end();
  }
}
