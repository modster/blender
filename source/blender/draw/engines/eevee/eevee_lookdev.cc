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

#include "NOD_shader.h"

#include "BKE_image.h"
#include "BKE_lib_id.h"
#include "BKE_node.h"
#include "BKE_studiolight.h"
#include "BKE_world.h"

#include "BLI_math_matrix.h"

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

void LookDev::init(void)
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
}

bool LookDev::sync_world(void)
{
  if (studiolight_ == nullptr) {
    return false;
  }
  /* World light probes render. */
  bNodeTree *nodetree = world_tree.nodetree_get(instensity_);
  GPUMaterial *gpumat = inst_.shaders.material_shader_get(
      "LookdevShader", material, nodetree, MAT_GEOM_WORLD, MAT_DOMAIN_SURFACE, true);

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

void LookDev::sync(void)
{
  if (studiolight_ == nullptr) {
    return;
  }
  /* Viewport display. */
  background_lookdev_ = DRW_pass_create("LookDev Background", DRW_STATE_WRITE_COLOR);

  GPUShader *sh = inst_.shaders.static_shader_get(LOOKDEV_BACKGROUND);
  DRWShadingGroup *grp = DRW_shgroup_create(sh, background_lookdev_);
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
  DRW_draw_pass(background_lookdev_);
  return true;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name LookDev Reference spheres
 *
 * Render reference spheres into a separate framebuffer to not distrub the main rendering.
 * The final texture is composited onto the render.
 * \{ */

/* Renders the reference spheres. This is similar to the main render_sample() of the instance. */
void LookDev::render_sample(void)
{
  const View3D *v3d = inst_.v3d;
  /* Only show the HDRI Preview in Shading Preview in the Viewport. */
  if (v3d == NULL || v3d->shading.type != OB_MATERIAL) {
    return;
  }
  /* Only show the HDRI Preview when viewing the Combined render pass */
  if (v3d->shading.render_pass != SCE_PASS_COMBINED) {
    return;
  }
  if (v3d->flag2 & V3D_HIDE_OVERLAYS) {
    return;
  }
  if ((v3d->overlay.flag & V3D_OVERLAY_LOOK_DEV) == 0) {
    return;
  }

  /* TODO(fclem) Do full rendering of the 2 spheres on a special film and composite
   * on top of main film. */
}

void LookDev::resolve_viewport(GPUFrameBuffer *UNUSED(default_fb))
{
  /* TODO(fclem). */
}

/** \} */

}  // namespace blender::eevee
