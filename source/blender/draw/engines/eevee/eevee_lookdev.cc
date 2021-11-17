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
 * Copyright 2018, Blender Foundation.
 */

/** \file
 * \ingroup eevee
 */

#include "BKE_image.h"
#include "BKE_lib_id.h"
#include "BKE_node.h"
#include "BKE_studiolight.h"
#include "BKE_world.h"
#include "BLI_math_matrix.h"
#include "BLI_rect.h"
#include "DNA_userdef_types.h"
#include "ED_screen.h"
#include "NOD_shader.h"

#include "eevee_instance.hh"

namespace blender::eevee {

/* -------------------------------------------------------------------- */
/** \name Lookdev Nodetree
 *
 * \{ */

LookDevWorldNodeTree::LookDevWorldNodeTree()
{
  bNodeTree *ntree = ntreeAddTree(NULL, "Lookdev Nodetree", ntreeType_Shader->idname);
  bNode *background = nodeAddStaticNode(NULL, ntree, SH_NODE_BACKGROUND);
  bNode *output = nodeAddStaticNode(NULL, ntree, SH_NODE_OUTPUT_WORLD);
  bNodeSocket *background_out = nodeFindSocket(background, SOCK_OUT, "Background");
  bNodeSocket *output_in = nodeFindSocket(output, SOCK_IN, "Surface");
  nodeAddLink(ntree, background, background_out, output, output_in);
  nodeSetActive(ntree, output);

  /* Note that we do not populate the environment texture input.
   * We plug the GPUTexture directly using the sampler binding name ("samp1"). */
  bNode *environment = nodeAddStaticNode(NULL, ntree, SH_NODE_TEX_ENVIRONMENT);
  bNodeSocket *background_in = nodeFindSocket(background, SOCK_IN, "Color");
  bNodeSocket *environment_out = nodeFindSocket(environment, SOCK_OUT, "Color");
  nodeAddLink(ntree, environment, environment_out, background, background_in);

  strength_socket_ =
      (bNodeSocketValueFloat *)nodeFindSocket(background, SOCK_IN, "Strength")->default_value;

  ntree_ = ntree;
}

LookDevWorldNodeTree::~LookDevWorldNodeTree()
{
  ntreeFreeEmbeddedTree(ntree_);
  MEM_SAFE_FREE(ntree_);
}

/* Configure a default nodetree with the given parameters. */
bNodeTree *LookDevWorldNodeTree::nodetree_get(float strength)
{
  /* WARNING: This function is not threadsafe. Which is not a problem for the moment. */
  strength_socket_->value = strength;
  return ntree_;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name LookDev Studiolight
 *
 * Light the scene using the studiolight hdri. Overrides the lightcache (if any) and
 * use custom shader to draw the background.
 * \{ */

void LookDev::init(const ivec2 &output_res, const rcti *render_border)
{
  StudioLight *studiolight = nullptr;
  if (inst_.v3d) {
    studiolight = BKE_studiolight_find(inst_.v3d->shading.lookdev_light,
                                       STUDIOLIGHT_ORIENTATIONS_MATERIAL_MODE);
  }

  if (inst_.use_studio_light() && studiolight && (studiolight->flag & STUDIOLIGHT_TYPE_WORLD)) {
    const View3DShading &shading = inst_.v3d->shading;
    studiolight_ = studiolight;

    /* Detect update. */
    if ((opacity_ != shading.studiolight_background) || (rotation_ != shading.studiolight_rot_z) ||
        (instensity_ != shading.studiolight_intensity) || (blur_ != shading.studiolight_blur) ||
        (view_rotation_ != ((shading.flag & V3D_SHADING_STUDIOLIGHT_VIEW_ROTATION) != 0)) ||
        (studiolight_index_ != studiolight_->index)) {
      opacity_ = shading.studiolight_background;
      instensity_ = shading.studiolight_intensity;
      blur_ = shading.studiolight_blur;
      rotation_ = shading.studiolight_rot_z;
      studiolight_index_ = studiolight_->index;
      view_rotation_ = (shading.flag & V3D_SHADING_STUDIOLIGHT_VIEW_ROTATION) != 0;

      inst_.sampling.reset();
      inst_.lightprobes.set_world_dirty();

      /* Update the material. */
      GPU_material_free(&material);
    }
  }
  else {
    if (studiolight_ != nullptr) {
      inst_.sampling.reset();
      inst_.lightprobes.set_world_dirty();
    }
    studiolight_ = nullptr;
    studiolight_index_ = -1;

    GPU_material_free(&material);
  }

  if (do_overlay(output_res, render_border)) {
    rcti rect;
    if (DRW_state_is_opengl_render()) {
      BLI_rcti_init(&rect, 0, output_res.x, 0, output_res.y);
    }
    else {
      const DRWContextState *draw_ctx = DRW_context_state_get();
      rect = *ED_region_visible_rect(draw_ctx->region);
    }

    /* Make the viewport width scale the lookdev spheres a bit.
     * Scale between 1000px and 2000px. */
    float viewport_scale = clamp_f(BLI_rcti_size_x(&rect) / (2000.0f * U.dpi_fac), 0.5f, 1.0f);
    int sphere_size = U.lookdev_sphere_size * U.dpi_fac * viewport_scale;
    ivec2 anchor = ivec2(rect.xmax, rect.ymin);

    if (sphere_size != sphere_size_ || anchor != anchor_) {
      /* Make sphere resolution adaptive to viewport_scale, dpi and lookdev_sphere_size */
      float res_scale = (U.lookdev_sphere_size / 400.0f) * viewport_scale * U.dpi_fac;
      if (res_scale > 0.7f) {
        sphere_lod_ = DRW_LOD_HIGH;
      }
      else if (res_scale > 0.25f) {
        sphere_lod_ = DRW_LOD_MEDIUM;
      }
      else {
        sphere_lod_ = DRW_LOD_LOW;
      }
      sphere_size_ = sphere_size;
      anchor_ = anchor;
      inst_.sampling.reset();
    }
  }
  else if (sphere_size_ != 0) {
    sphere_size_ = 0;
    inst_.sampling.reset();
  }
}

bool LookDev::do_overlay(const ivec2 &output_res, const rcti *render_border)
{
  const View3D *v3d = inst_.v3d;
  /* Only show the HDRI Preview in Shading Preview in the Viewport. */
  if (v3d == nullptr || v3d->shading.type != OB_MATERIAL) {
    return false;
  }
  /* Only show the HDRI Preview when viewing the Combined render pass */
  if (v3d->shading.render_pass != SCE_PASS_COMBINED) {
    return false;
  }
  if (v3d->flag2 & V3D_HIDE_OVERLAYS) {
    return false;
  }
  if ((v3d->overlay.flag & V3D_OVERLAY_LOOK_DEV) == 0) {
    return false;
  }
  if (inst_.camera.is_panoramic()) {
    return false;
  }
  if (output_res != ivec2(BLI_rcti_size_x(render_border), BLI_rcti_size_y(render_border))) {
    /* TODO(fclem) support this case. */
    return false;
  }
  return true;
}

bool LookDev::sync_world(void)
{
  if (studiolight_ == nullptr) {
    return false;
  }
  /* World light probes render. */
  bNodeTree *nodetree = world_tree.nodetree_get(instensity_);
  GPUMaterial *gpumat = inst_.shaders.material_shader_get(
      "LookDev", material, nodetree, MAT_PIPE_FORWARD, MAT_GEOM_WORLD, true);

  BKE_studiolight_ensure_flag(studiolight_, STUDIOLIGHT_EQUIRECT_RADIANCE_GPUTEXTURE);
  GPUTexture *gputex = studiolight_->equirect_radiance_gputexture;

  if (gputex == nullptr) {
    return false;
  }
  inst_.shading_passes.background.sync(gpumat, gputex);
  return true;
}

void LookDev::rotation_get(mat4 r_mat)
{
  if (studiolight_ == nullptr) {
    unit_m4(r_mat);
  }
  else {
    axis_angle_to_mat4_single(r_mat, 'Z', rotation_);
  }

  if (view_rotation_) {
    float x_rot_matrix[4][4];
    const CameraData &cam = inst_.camera.data_get();
    axis_angle_to_mat4_single(x_rot_matrix, 'X', M_PI / 2.0f);
    mul_m4_m4m4(x_rot_matrix, x_rot_matrix, cam.viewmat);
    mul_m4_m4m4(r_mat, r_mat, x_rot_matrix);
  }
}

void LookDev::sync_background(void)
{
  if (studiolight_ == nullptr) {
    return;
  }
  /* Viewport display. */
  background_ps_ = DRW_pass_create("LookDev.Background", DRW_STATE_WRITE_COLOR);

  GPUShader *sh = inst_.shaders.static_shader_get(LOOKDEV_BACKGROUND);
  DRWShadingGroup *grp = DRW_shgroup_create(sh, background_ps_);
  DRW_shgroup_uniform_texture_ref(grp, "lightprobe_cube_tx", inst_.lightprobes.cube_tx_ref_get());
  DRW_shgroup_uniform_block(grp, "lightprobes_info_block", inst_.lightprobes.info_ubo_get());
  DRW_shgroup_uniform_float_copy(grp, "blur", clamp_f(blur_, 0.0f, 0.99999f));
  DRW_shgroup_uniform_float_copy(grp, "opacity", opacity_);
  DRW_shgroup_call_procedural_triangles(grp, nullptr, 1);
}

/* Renders background using lightcache. */
bool LookDev::render_background(void)
{
  if (studiolight_ == nullptr) {
    return false;
  }
  DRW_draw_pass(background_ps_);
  return true;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name LookDev Reference spheres
 *
 * Render reference spheres into a separate framebuffer to not distrub the main rendering.
 * The final texture is composited onto the render.
 * \{ */

void LookDev::sync_overlay(void)
{
  if (sphere_size_ == 0) {
    return;
  }

  DRWState state = DRW_STATE_WRITE_COLOR | DRW_STATE_WRITE_DEPTH | DRW_STATE_DEPTH_ALWAYS |
                   DRW_STATE_CULL_BACK;
  overlay_ps_ = DRW_pass_create("LookDev.Overlay", state);

  GPUBatch *sphere = DRW_cache_sphere_get(sphere_lod_);

  const CameraData &cam = inst_.camera.data_get();
  LightModule &lights = inst_.lights;
  LightProbeModule &lightprobes = inst_.lightprobes;

  /* Jitter for AA. */
  vec2 jitter = -0.5f + vec2(inst_.sampling.rng_get(SAMPLING_FILTER_U),
                             inst_.sampling.rng_get(SAMPLING_FILTER_V));

  /* Matrix used to position the spheres in viewport space. */
  mat4 sphere_mat;
  copy_m4_m4(sphere_mat, cam.viewmat);

  const float *viewport_size = DRW_viewport_size_get();
  const int sphere_margin = sphere_size_ / 6;
  vec2 offset = vec2(0, sphere_margin);

  std::array<::Material *, 2> materials = {inst_.materials.diffuse_mat_,
                                           inst_.materials.glossy_mat_};
  for (::Material *mat : materials) {
    GPUMaterial *gpumat = inst_.shaders.material_shader_get(
        mat, mat->nodetree, MAT_PIPE_FORWARD, MAT_GEOM_LOOKDEV, false);
    DRWShadingGroup *grp = DRW_shgroup_material_create(gpumat, overlay_ps_);
    DRW_shgroup_uniform_block_ref(grp, "lights_block", lights.lights_ubo_ref_get());
    DRW_shgroup_uniform_block_ref(grp, "shadows_block", lights.shadows_ubo_ref_get());
    DRW_shgroup_uniform_block_ref(grp, "lights_culling_block", lights.culling_ubo_ref_get());
    DRW_shgroup_uniform_block(grp, "sampling_block", inst_.sampling.ubo_get());
    DRW_shgroup_uniform_block(grp, "grids_block", lightprobes.grid_ubo_get());
    DRW_shgroup_uniform_block(grp, "cubes_block", lightprobes.cube_ubo_get());
    DRW_shgroup_uniform_block(grp, "lightprobes_info_block", lightprobes.info_ubo_get());
    DRW_shgroup_uniform_texture_ref(grp, "lightprobe_grid_tx", lightprobes.grid_tx_ref_get());
    DRW_shgroup_uniform_texture_ref(grp, "lightprobe_cube_tx", lightprobes.cube_tx_ref_get());
    DRW_shgroup_uniform_texture_ref(grp, "lights_culling_tx", lights.culling_tx_ref_get());
    DRW_shgroup_uniform_texture(grp, "utility_tx", inst_.shading_passes.utility_tx);
    DRW_shgroup_uniform_texture(grp, "shadow_atlas_tx", inst_.shadows.atlas_tx_get());

    offset.x -= sphere_size_ + sphere_margin;

    /* Pass 2D scale and bias factor in the last column. */
    vec2 scale = sphere_size_ / vec2(viewport_size);
    vec2 bias = -1.0f + scale + 2.0f * (anchor_ + offset + jitter) / vec2(viewport_size);
    copy_v4_fl4(sphere_mat[3], UNPACK2(scale), UNPACK2(bias));
    DRW_shgroup_call_obmat(grp, sphere, sphere_mat);

    offset.x -= sphere_margin;
  }

  view_ = nullptr;
}

/* Renders the reference spheres. */
void LookDev::render_overlay(GPUFrameBuffer *fb)
{
  if (sphere_size_ == 0) {
    return;
  }

  const DRWView *active_view = DRW_view_get_active();

  inst_.lightprobes.set_view(active_view, ivec2(0));
  inst_.lights.set_view(active_view, ivec2(0));
  inst_.lights.bind_batch(0);

  /* Create subview for correct shading. Sub because we don not care about culling. */
  const CameraData &cam = inst_.camera.data_get();
  mat4 winmat;
  orthographic_m4(winmat, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f);
  if (view_) {
    DRW_view_update_sub(view_, cam.viewmat, winmat);
  }
  else {
    view_ = DRW_view_create_sub(active_view, cam.viewmat, winmat);
  }

  DRW_view_set_active(view_);

  GPU_framebuffer_bind(fb);
  DRW_draw_pass(overlay_ps_);

  DRW_view_set_active(active_view);
}

/** \} */

}  // namespace blender::eevee
