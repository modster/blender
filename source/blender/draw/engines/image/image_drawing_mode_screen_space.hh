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
 * \ingroup draw_engine
 */

#pragma once

#include "image_private.hh"

#include "BKE_image_partial_update.hh"

namespace blender::draw::image_engine {

using namespace blender::bke::image::partial_update;

/* TODO: Should we use static class functions in stead of a namespace. */
namespace clipping {

/** \brief Update the texture slot uv and screen space bounds. */
static void update_texture_slots_bounds(const ARegion *region,
                                        const AbstractSpaceAccessor *space,
                                        IMAGE_PrivateData *pd)
{
  // each texture
  BLI_rctf_init(
      &pd->screen_space.texture_infos[0].clipping_bounds, 0, region->winx, 0, region->winy);
  // TODO: calculate the correct visible uv bounds.
  BLI_rctf_init(&pd->screen_space.texture_infos[0].uv_bounds, 0.0, 1.0, 0.0, 1.0);

  /* Mark the other textures as invalid. */
  for (int i = 1; i < SCREEN_SPACE_DRAWING_MODE_TEXTURE_LEN; i++) {
    BLI_rctf_init_minmax(&pd->screen_space.texture_infos[i].clipping_bounds);
    BLI_rctf_init_minmax(&pd->screen_space.texture_infos[i].uv_bounds);
  }
}

static void update_texture_slots_visibility(const AbstractSpaceAccessor *space,
                                            IMAGE_PrivateData *pd)
{
  pd->screen_space.texture_infos[0].visible = true;
  for (int i = 1; i < SCREEN_SPACE_DRAWING_MODE_TEXTURE_LEN; i++) {
    pd->screen_space.texture_infos[i].visible = false;
  }
}

}  // namespace clipping

class ScreenSpaceDrawingMode : public AbstractDrawingMode {
 private:
  DRWPass *create_image_pass() const
  {
    /* Write depth is needed for background overlay rendering. Near depth is used for
     * transparency checker and Far depth is used for indicating the image size. */
    DRWState state = static_cast<DRWState>(DRW_STATE_WRITE_COLOR | DRW_STATE_WRITE_DEPTH |
                                           DRW_STATE_DEPTH_ALWAYS | DRW_STATE_BLEND_ALPHA_PREMUL);
    return DRW_pass_create("Image", state);
  }

  void add_shgroups(IMAGE_PassList *psl,
                    IMAGE_TextureList *txl,
                    IMAGE_PrivateData *pd,
                    const ShaderParameters &sh_params) const
  {
    GPUBatch *geom = DRW_cache_quad_get();
    GPUShader *shader = IMAGE_shader_image_get(false);

    float image_mat[4][4];
    unit_m4(image_mat);
    for (int i = 0; i < SCREEN_SPACE_DRAWING_MODE_TEXTURE_LEN; i++) {
      if (!pd->screen_space.texture_infos[i].visible) {
        continue;
      }

      image_mat[0][0] = pd->screen_space.texture_infos[i].clipping_bounds.xmax;
      image_mat[1][1] = pd->screen_space.texture_infos[i].clipping_bounds.ymax;

      // TODO: use subgroup.
      DRWShadingGroup *shgrp = DRW_shgroup_create(shader, psl->image_pass);
      DRW_shgroup_uniform_texture_ex(
          shgrp, "imageTexture", txl->screen_space.textures[i], GPU_SAMPLER_DEFAULT);
      DRW_shgroup_uniform_vec2_copy(shgrp, "farNearDistances", sh_params.far_near);
      DRW_shgroup_uniform_vec4_copy(shgrp, "color", ShaderParameters::color);
      DRW_shgroup_uniform_vec4_copy(shgrp, "shuffle", sh_params.shuffle);
      DRW_shgroup_uniform_int_copy(shgrp, "drawFlags", sh_params.flags);
      DRW_shgroup_uniform_bool_copy(shgrp, "imgPremultiplied", sh_params.use_premul_alpha);

      DRW_shgroup_call_obmat(shgrp, geom, image_mat);
    }
  }

  /**
   * \brief check if the partial update user in the private data can still be used.
   *
   * When switching to a different image the partial update user should be recreated.
   */
  bool partial_update_is_valid(const IMAGE_PrivateData *pd, const Image *image) const
  {
    if (pd->screen_space.partial_update_image != image) {
      return false;
    }

    return pd->screen_space.partial_update_user != nullptr;
  }

  void partial_update_allocate(IMAGE_PrivateData *pd, const Image *image) const
  {
    BLI_assert(pd->screen_space.partial_update_user == nullptr);
    pd->screen_space.partial_update_user = BKE_image_partial_update_create(image);
    pd->screen_space.partial_update_image = image;
  }

  void partial_update_free(IMAGE_PrivateData *pd) const
  {
    if (pd->screen_space.partial_update_user != nullptr) {
      BKE_image_partial_update_free(pd->screen_space.partial_update_user);
      pd->screen_space.partial_update_user = nullptr;
    }
  }

  void update_texture_slot_allocation(IMAGE_TextureList *txl, IMAGE_PrivateData *pd) const
  {
    for (int i = 0; i < SCREEN_SPACE_DRAWING_MODE_TEXTURE_LEN; i++) {
      const bool is_allocated = txl->screen_space.textures[i] != nullptr;
      const bool is_visible = pd->screen_space.texture_infos[i].visible;
      const bool should_be_freed = !is_visible && is_allocated;
      const bool should_be_created = is_visible && !is_allocated;

      if (should_be_freed) {
        GPU_texture_free(txl->screen_space.textures[i]);
        txl->screen_space.textures[i] = nullptr;
      }

      if (should_be_created) {
        DRW_texture_ensure_fullscreen_2d(
            &txl->screen_space.textures[i], GPU_RGBA16F, static_cast<DRWTextureFlag>(0));
      }
      pd->screen_space.texture_infos[i].dirty = should_be_created;
    }
  }

  void mark_all_texture_slots_dirty(IMAGE_PrivateData *pd) const
  {
    for (int i = 0; i < SCREEN_SPACE_DRAWING_MODE_TEXTURE_LEN; i++) {
      pd->screen_space.texture_infos[i].dirty = true;
    }
  }

  void update_textures(IMAGE_TextureList *txl,
                       IMAGE_PrivateData *pd,
                       Image *image,
                       ImageUser *image_user) const
  {
    PartialUpdateChecker<ImageTileData> checker(
        image, image_user, pd->screen_space.partial_update_user);
    PartialUpdateChecker<ImageTileData>::CollectResult changes = checker.collect_changes();

    switch (changes.get_result_code()) {
      case ePartialUpdateCollectResult::FullUpdateNeeded:
        mark_all_texture_slots_dirty(pd);
        break;
      case ePartialUpdateCollectResult::NoChangesDetected:
        break;
      case ePartialUpdateCollectResult::PartialChangesDetected:
        do_partial_update(changes, txl, pd, image);
        break;
    }
    update_dirty_textures(txl, pd);
  }

  void do_partial_update(PartialUpdateChecker<ImageTileData>::CollectResult &iterator,
                         IMAGE_TextureList *txl,
                         IMAGE_PrivateData *pd,
                         Image *image) const
  {
    while (iterator.get_next_change() == ePartialUpdateIterResult::ChangeAvailable) {
      for (int i = 0; i < SCREEN_SPACE_DRAWING_MODE_TEXTURE_LEN; i++) {
        /* Dirty images will receive a full update. No need to do a partial one now. */
        if (pd->screen_space.texture_infos[i].dirty) {
          continue;
        }
        // TODO
      }
    }
  }

  void update_dirty_textures(IMAGE_TextureList *txl, IMAGE_PrivateData *pd) const
  {
    for (int i = 0; i < SCREEN_SPACE_DRAWING_MODE_TEXTURE_LEN; i++) {
      if (!pd->screen_space.texture_infos[i].dirty) {
        continue;
      }
      if (!pd->screen_space.texture_infos[i].visible) {
        continue;
      }

      rctf *uv_bounds = &pd->screen_space.texture_infos[i].uv_bounds;

      GPUTexture *gpu_texture = txl->screen_space.textures[i];
      int texture_width = GPU_texture_width(gpu_texture);
      int texture_height = GPU_texture_height(gpu_texture);
      int texture_size = texture_width * texture_height;

      float *data = static_cast<float *>(MEM_mallocN(sizeof(float) * 4 * texture_size, __func__));
      int offset = 0;
      for (int y = 0; y < texture_height; y++) {
        float v = y / (float)texture_height;
        for (int x = 0; x < texture_width; x++) {
          float u = x / (float)texture_width;
          copy_v4_fl4(&data[offset * 4],
                      uv_bounds->xmin * u + uv_bounds->xmax * (1.0 - u),
                      uv_bounds->ymin * v + uv_bounds->ymax * (1.0 - v),
                      0.0,
                      1.0);
          offset++;
        }
      }
      GPU_texture_update(gpu_texture, GPU_DATA_FLOAT, data);

      MEM_freeN(data);
    }
  }

 public:
  void cache_init(IMAGE_Data *vedata) const override
  {
    IMAGE_PassList *psl = vedata->psl;

    psl->image_pass = create_image_pass();
  }

  void cache_image(AbstractSpaceAccessor *space,
                   IMAGE_Data *vedata,
                   Image *image,
                   ImageUser *iuser,
                   ImBuf *image_buffer) const override
  {
    const DRWContextState *draw_ctx = DRW_context_state_get();
    IMAGE_PassList *psl = vedata->psl;
    IMAGE_TextureList *txl = vedata->txl;
    IMAGE_StorageList *stl = vedata->stl;
    IMAGE_PrivateData *pd = stl->pd;

    if (!partial_update_is_valid(pd, image)) {
      partial_update_free(pd);
      partial_update_allocate(pd, image);
    }

    // Step: Find out which screen space textures are needed to draw on the screen. Remove the
    // screen space textures that aren't needed.
    const ARegion *region = draw_ctx->region;
    clipping::update_texture_slots_bounds(region, space, pd);
    clipping::update_texture_slots_visibility(space, pd);
    update_texture_slot_allocation(txl, pd);

    // Step: Update the GPU textures based on the changes in the image.
    update_textures(txl, pd, image, iuser);

    // Step: Add the GPU textures to the shgroup.

    GPUTexture *tex_tile_data = nullptr;
    space->get_gpu_textures(
        image, iuser, image_buffer, &pd->texture, &pd->owns_texture, &tex_tile_data);
    if (pd->texture == nullptr) {
      return;
    }
    const bool is_tiled_texture = tex_tile_data != nullptr;

    ShaderParameters sh_params;
    sh_params.use_premul_alpha = BKE_image_has_gpu_texture_premultiplied_alpha(image,
                                                                               image_buffer);

    const Scene *scene = draw_ctx->scene;
    if (scene->camera && scene->camera->type == OB_CAMERA) {
      Camera *camera = static_cast<Camera *>(scene->camera->data);
      copy_v2_fl2(sh_params.far_near, camera->clip_end, camera->clip_start);
    }
    space->get_shader_parameters(sh_params, image_buffer, is_tiled_texture);

    add_shgroups(psl, txl, pd, sh_params);
  }

  void draw_finish(IMAGE_Data *vedata) const override
  {
    IMAGE_StorageList *stl = vedata->stl;
    IMAGE_PrivateData *pd = stl->pd;

    if (pd->texture && pd->owns_texture) {
      GPU_texture_free(pd->texture);
      pd->owns_texture = false;
    }
    pd->texture = nullptr;
  }

  void draw_scene(IMAGE_Data *vedata) const override
  {
    IMAGE_PassList *psl = vedata->psl;
    IMAGE_PrivateData *pd = vedata->stl->pd;

    DefaultFramebufferList *dfbl = DRW_viewport_framebuffer_list_get();
    GPU_framebuffer_bind(dfbl->default_fb);
    static float clear_col[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    GPU_framebuffer_clear_color_depth(dfbl->default_fb, clear_col, 1.0);

    DRW_view_set_active(pd->view);
    DRW_draw_pass(psl->image_pass);
    DRW_view_set_active(nullptr);
  }
};  // namespace clipping

}  // namespace blender::draw::image_engine
