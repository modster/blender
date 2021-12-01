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
 * Copyright 2019, Blender Foundation.
 */

/** \file
 * \ingroup draw_engine
 */

#include "DRW_render.h"

#include "UI_resources.h"

#include "BKE_anim_path.h"
#include "BKE_camera.h"
#include "BKE_constraint.h"
#include "BKE_curve.h"
#include "BKE_global.h"
#include "BKE_mball.h"
#include "BKE_mesh.h"
#include "BKE_modifier.h"
#include "BKE_movieclip.h"
#include "BKE_object.h"
#include "BKE_rigidbody.h"
#include "BKE_tracking.h"

#include "BLI_listbase.h"
#include "BLI_task.h"

#include "DNA_camera_types.h"
#include "DNA_constraint_types.h"
#include "DNA_curve_types.h"
#include "DNA_fluid_types.h"
#include "DNA_lightprobe_types.h"
#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"
#include "DNA_meta_types.h"
#include "DNA_modifier_types.h"
#include "DNA_pointcache_types.h"
#include "DNA_rigidbody_types.h"

#include "DEG_depsgraph_query.h"

#include "ED_view3d.h"

#include "GPU_batch.h"

#include "overlay_private.h"

#include "draw_cache_impl.h"
#include "draw_common.h"
#include "draw_manager_text.h"

#include "BLI_math_rotation.h"
#include <math.h>

#ifdef WITH_BULLET
#  include "RBI_api.h"
#endif

void OVERLAY_extra_cache_init(OVERLAY_Data *vedata)
{
  OVERLAY_PassList *psl = vedata->psl;
  OVERLAY_TextureList *txl = vedata->txl;
  OVERLAY_PrivateData *pd = vedata->stl->pd;
  const bool is_select = DRW_state_is_select();

  DRWState state_blend = DRW_STATE_WRITE_COLOR | DRW_STATE_BLEND_ALPHA;
  DRW_PASS_CREATE(psl->extra_blend_ps, state_blend | pd->clipping_state);
  DRW_PASS_CREATE(psl->extra_centers_ps, state_blend | pd->clipping_state);

  {
    DRWState state = DRW_STATE_WRITE_COLOR;

    DRW_PASS_CREATE(psl->extra_grid_ps, state | pd->clipping_state);
    DefaultTextureList *dtxl = DRW_viewport_texture_list_get();
    DRWShadingGroup *grp;
    struct GPUShader *sh = OVERLAY_shader_extra_grid();
    struct GPUTexture *tex = DRW_state_is_fbo() ? dtxl->depth : txl->dummy_depth_tx;

    pd->extra_grid_grp = grp = DRW_shgroup_create(sh, psl->extra_grid_ps);
    DRW_shgroup_uniform_texture(grp, "depthBuffer", tex);
    DRW_shgroup_uniform_block(grp, "globalsBlock", G_draw.block_ubo);
    DRW_shgroup_uniform_bool_copy(grp, "isTransform", (G.moving & G_TRANSFORM_OBJ) != 0);
  }

  for (int i = 0; i < 2; i++) {
    /* Non Meshes Pass (Camera, empties, lights ...) */
    struct GPUShader *sh;
    struct GPUVertFormat *format;
    DRWShadingGroup *grp, *grp_sub;

    OVERLAY_InstanceFormats *formats = OVERLAY_shader_instance_formats_get();
    OVERLAY_ExtraCallBuffers *cb = &pd->extra_call_buffers[i];
    DRWPass **p_extra_ps = &psl->extra_ps[i];

    DRWState infront_state = (DRW_state_is_select() && (i == 1)) ? DRW_STATE_IN_FRONT_SELECT : 0;
    DRWState state = DRW_STATE_WRITE_COLOR | DRW_STATE_WRITE_DEPTH | DRW_STATE_DEPTH_LESS_EQUAL;
    DRW_PASS_CREATE(*p_extra_ps, state | pd->clipping_state | infront_state);

    DRWPass *extra_ps = *p_extra_ps;

#define BUF_INSTANCE DRW_shgroup_call_buffer_instance
#define BUF_POINT(grp, format) DRW_shgroup_call_buffer(grp, format, GPU_PRIM_POINTS)
#define BUF_LINE(grp, format) DRW_shgroup_call_buffer(grp, format, GPU_PRIM_LINES)

    /* Sorted by shader to avoid state changes during render. */
    {
      format = formats->instance_extra;
      sh = OVERLAY_shader_extra(is_select);

      grp = DRW_shgroup_create(sh, extra_ps);
      DRW_shgroup_uniform_block(grp, "globalsBlock", G_draw.block_ubo);

      grp_sub = DRW_shgroup_create_sub(grp);
      cb->camera_distances = BUF_INSTANCE(grp_sub, format, DRW_cache_camera_distances_get());
      cb->camera_frame = BUF_INSTANCE(grp_sub, format, DRW_cache_camera_frame_get());
      cb->camera_tria[0] = BUF_INSTANCE(grp_sub, format, DRW_cache_camera_tria_wire_get());
      cb->camera_tria[1] = BUF_INSTANCE(grp_sub, format, DRW_cache_camera_tria_get());
      cb->empty_axes = BUF_INSTANCE(grp_sub, format, DRW_cache_bone_arrows_get());
      cb->empty_capsule_body = BUF_INSTANCE(grp_sub, format, DRW_cache_empty_capsule_body_get());
      cb->empty_capsule_cap = BUF_INSTANCE(grp_sub, format, DRW_cache_empty_capsule_cap_get());
      cb->empty_circle = BUF_INSTANCE(grp_sub, format, DRW_cache_circle_get());
      cb->empty_cone = BUF_INSTANCE(grp_sub, format, DRW_cache_empty_cone_get());
      cb->empty_cube = BUF_INSTANCE(grp_sub, format, DRW_cache_empty_cube_get());
      cb->empty_cylinder = BUF_INSTANCE(grp_sub, format, DRW_cache_empty_cylinder_get());
      cb->empty_image_frame = BUF_INSTANCE(grp_sub, format, DRW_cache_quad_wires_get());
      cb->empty_plain_axes = BUF_INSTANCE(grp_sub, format, DRW_cache_plain_axes_get());
      cb->empty_single_arrow = BUF_INSTANCE(grp_sub, format, DRW_cache_single_arrow_get());
      cb->empty_sphere = BUF_INSTANCE(grp_sub, format, DRW_cache_empty_sphere_get());
      cb->empty_sphere_solid = BUF_INSTANCE(grp_sub, format, DRW_cache_sphere_get(DRW_LOD_LOW));
      cb->field_cone_limit = BUF_INSTANCE(grp_sub, format, DRW_cache_field_cone_limit_get());
      cb->field_curve = BUF_INSTANCE(grp_sub, format, DRW_cache_field_curve_get());
      cb->field_force = BUF_INSTANCE(grp_sub, format, DRW_cache_field_force_get());
      cb->field_sphere_limit = BUF_INSTANCE(grp_sub, format, DRW_cache_field_sphere_limit_get());
      cb->field_tube_limit = BUF_INSTANCE(grp_sub, format, DRW_cache_field_tube_limit_get());
      cb->field_vortex = BUF_INSTANCE(grp_sub, format, DRW_cache_field_vortex_get());
      cb->field_wind = BUF_INSTANCE(grp_sub, format, DRW_cache_field_wind_get());
      cb->light_area[0] = BUF_INSTANCE(grp_sub, format, DRW_cache_light_area_disk_lines_get());
      cb->light_area[1] = BUF_INSTANCE(grp_sub, format, DRW_cache_light_area_square_lines_get());
      cb->light_point = BUF_INSTANCE(grp_sub, format, DRW_cache_light_point_lines_get());
      cb->light_spot = BUF_INSTANCE(grp_sub, format, DRW_cache_light_spot_lines_get());
      cb->light_sun = BUF_INSTANCE(grp_sub, format, DRW_cache_light_sun_lines_get());
      cb->probe_cube = BUF_INSTANCE(grp_sub, format, DRW_cache_lightprobe_cube_get());
      cb->probe_grid = BUF_INSTANCE(grp_sub, format, DRW_cache_lightprobe_grid_get());
      cb->probe_planar = BUF_INSTANCE(grp_sub, format, DRW_cache_lightprobe_planar_get());
      cb->solid_quad = BUF_INSTANCE(grp_sub, format, DRW_cache_quad_get());
      cb->speaker = BUF_INSTANCE(grp_sub, format, DRW_cache_speaker_get());

      grp_sub = DRW_shgroup_create_sub(grp);
      DRW_shgroup_state_enable(grp_sub, DRW_STATE_DEPTH_ALWAYS);
      DRW_shgroup_state_disable(grp_sub, DRW_STATE_DEPTH_LESS_EQUAL);
      cb->origin_xform = BUF_INSTANCE(grp_sub, format, DRW_cache_bone_arrows_get());
    }
    {
      format = formats->instance_extra;
      grp = DRW_shgroup_create(sh, psl->extra_blend_ps); /* NOTE: not the same pass! */
      DRW_shgroup_uniform_block(grp, "globalsBlock", G_draw.block_ubo);

      grp_sub = DRW_shgroup_create_sub(grp);
      DRW_shgroup_state_enable(grp_sub, DRW_STATE_DEPTH_LESS_EQUAL | DRW_STATE_CULL_BACK);
      cb->camera_volume = BUF_INSTANCE(grp_sub, format, DRW_cache_camera_volume_get());
      cb->camera_volume_frame = BUF_INSTANCE(grp_sub, format, DRW_cache_camera_volume_wire_get());
      cb->light_spot_cone_back = BUF_INSTANCE(grp_sub, format, DRW_cache_light_spot_volume_get());

      grp_sub = DRW_shgroup_create_sub(grp);
      DRW_shgroup_state_enable(grp_sub, DRW_STATE_DEPTH_LESS_EQUAL | DRW_STATE_CULL_FRONT);
      cb->light_spot_cone_front = BUF_INSTANCE(grp_sub, format, DRW_cache_light_spot_volume_get());
    }
    {
      format = formats->instance_pos;
      sh = OVERLAY_shader_extra_groundline();

      grp = DRW_shgroup_create(sh, extra_ps);
      DRW_shgroup_uniform_block(grp, "globalsBlock", G_draw.block_ubo);
      DRW_shgroup_state_enable(grp, DRW_STATE_BLEND_ALPHA);

      cb->groundline = BUF_INSTANCE(grp, format, DRW_cache_groundline_get());
    }
    {
      sh = OVERLAY_shader_extra_wire(false, is_select);

      grp = DRW_shgroup_create(sh, extra_ps);
      DRW_shgroup_uniform_block(grp, "globalsBlock", G_draw.block_ubo);

      cb->extra_dashed_lines = BUF_LINE(grp, formats->pos_color);
      cb->extra_lines = BUF_LINE(grp, formats->wire_extra);
    }
    {
      sh = OVERLAY_shader_extra_wire(true, is_select);

      cb->extra_wire = grp = DRW_shgroup_create(sh, extra_ps);
      DRW_shgroup_uniform_block(grp, "globalsBlock", G_draw.block_ubo);
    }
    {
      sh = OVERLAY_shader_extra_loose_point();

      cb->extra_loose_points = grp = DRW_shgroup_create(sh, extra_ps);
      DRW_shgroup_uniform_block(grp, "globalsBlock", G_draw.block_ubo);

      /* Buffer access for drawing isolated points, matching `extra_lines`. */
      cb->extra_points = BUF_POINT(grp, formats->point_extra);
    }
    {
      format = formats->pos;
      sh = OVERLAY_shader_extra_point();

      grp = DRW_shgroup_create(sh, psl->extra_centers_ps); /* NOTE: not the same pass! */
      DRW_shgroup_uniform_block(grp, "globalsBlock", G_draw.block_ubo);

      grp_sub = DRW_shgroup_create_sub(grp);
      DRW_shgroup_uniform_vec4_copy(grp_sub, "color", G_draw.block.colorActive);
      cb->center_active = BUF_POINT(grp_sub, format);

      grp_sub = DRW_shgroup_create_sub(grp);
      DRW_shgroup_uniform_vec4_copy(grp_sub, "color", G_draw.block.colorSelect);
      cb->center_selected = BUF_POINT(grp_sub, format);

      grp_sub = DRW_shgroup_create_sub(grp);
      DRW_shgroup_uniform_vec4_copy(grp_sub, "color", G_draw.block.colorDeselect);
      cb->center_deselected = BUF_POINT(grp_sub, format);

      grp_sub = DRW_shgroup_create_sub(grp);
      DRW_shgroup_uniform_vec4_copy(grp_sub, "color", G_draw.block.colorLibrarySelect);
      cb->center_selected_lib = BUF_POINT(grp_sub, format);

      grp_sub = DRW_shgroup_create_sub(grp);
      DRW_shgroup_uniform_vec4_copy(grp_sub, "color", G_draw.block.colorLibrary);
      cb->center_deselected_lib = BUF_POINT(grp_sub, format);
    }
  }
}

void OVERLAY_extra_point(OVERLAY_ExtraCallBuffers *cb, const float point[3], const float color[4])
{
  DRW_buffer_add_entry(cb->extra_points, point, color);
}

void OVERLAY_extra_line_dashed(OVERLAY_ExtraCallBuffers *cb,
                               const float start[3],
                               const float end[3],
                               const float color[4])
{
  DRW_buffer_add_entry(cb->extra_dashed_lines, end, color);
  DRW_buffer_add_entry(cb->extra_dashed_lines, start, color);
}

void OVERLAY_extra_line(OVERLAY_ExtraCallBuffers *cb,
                        const float start[3],
                        const float end[3],
                        const int color_id)
{
  DRW_buffer_add_entry(cb->extra_lines, start, &color_id);
  DRW_buffer_add_entry(cb->extra_lines, end, &color_id);
}

OVERLAY_ExtraCallBuffers *OVERLAY_extra_call_buffer_get(OVERLAY_Data *vedata, Object *ob)
{
  bool do_in_front = (ob->dtx & OB_DRAW_IN_FRONT) != 0;
  OVERLAY_PrivateData *pd = vedata->stl->pd;
  return &pd->extra_call_buffers[do_in_front];
}

void OVERLAY_extra_loose_points(OVERLAY_ExtraCallBuffers *cb,
                                struct GPUBatch *geom,
                                const float mat[4][4],
                                const float color[4])
{
  float draw_mat[4][4];
  pack_v4_in_mat4(draw_mat, mat, color);
  DRW_shgroup_call_obmat(cb->extra_loose_points, geom, draw_mat);
}

void OVERLAY_extra_wire(OVERLAY_ExtraCallBuffers *cb,
                        struct GPUBatch *geom,
                        const float mat[4][4],
                        const float color[4])
{
  float draw_mat[4][4];
  const float col[4] = {UNPACK3(color), 0.0f /* No stipples. */};
  pack_v4_in_mat4(draw_mat, mat, col);
  DRW_shgroup_call_obmat(cb->extra_wire, geom, draw_mat);
}

/* -------------------------------------------------------------------- */
/** \name Empties
 * \{ */

void OVERLAY_empty_shape(OVERLAY_ExtraCallBuffers *cb,
                         const float mat[4][4],
                         const float draw_size,
                         const char draw_type,
                         const float color[4])
{
  float instdata[4][4];
  pack_fl_in_mat4(instdata, mat, draw_size);

  switch (draw_type) {
    case OB_PLAINAXES:
      DRW_buffer_add_entry(cb->empty_plain_axes, color, instdata);
      break;
    case OB_SINGLE_ARROW:
      DRW_buffer_add_entry(cb->empty_single_arrow, color, instdata);
      break;
    case OB_CUBE:
      DRW_buffer_add_entry(cb->empty_cube, color, instdata);
      break;
    case OB_CIRCLE:
      DRW_buffer_add_entry(cb->empty_circle, color, instdata);
      break;
    case OB_EMPTY_SPHERE:
      DRW_buffer_add_entry(cb->empty_sphere, color, instdata);
      break;
    case OB_EMPTY_CONE:
      DRW_buffer_add_entry(cb->empty_cone, color, instdata);
      break;
    case OB_ARROWS:
      DRW_buffer_add_entry(cb->empty_axes, color, instdata);
      break;
    case OB_EMPTY_IMAGE:
      /* This only show the frame. See OVERLAY_image_empty_cache_populate() for the image. */
      DRW_buffer_add_entry(cb->empty_image_frame, color, instdata);
      break;
  }
}

void OVERLAY_empty_cache_populate(OVERLAY_Data *vedata, Object *ob)
{
  if (((ob->base_flag & BASE_FROM_DUPLI) != 0) && ((ob->transflag & OB_DUPLICOLLECTION) != 0) &&
      ob->instance_collection) {
    return;
  }

  OVERLAY_ExtraCallBuffers *cb = OVERLAY_extra_call_buffer_get(vedata, ob);
  const DRWContextState *draw_ctx = DRW_context_state_get();
  ViewLayer *view_layer = draw_ctx->view_layer;
  float *color;

  switch (ob->empty_drawtype) {
    case OB_PLAINAXES:
    case OB_SINGLE_ARROW:
    case OB_CUBE:
    case OB_CIRCLE:
    case OB_EMPTY_SPHERE:
    case OB_EMPTY_CONE:
    case OB_ARROWS:
      DRW_object_wire_theme_get(ob, view_layer, &color);
      OVERLAY_empty_shape(cb, ob->obmat, ob->empty_drawsize, ob->empty_drawtype, color);
      break;
    case OB_EMPTY_IMAGE:
      OVERLAY_image_empty_cache_populate(vedata, ob);
      break;
  }
}


static void OVERLAY_non_primitive_collision_shape(OVERLAY_ExtraCallBuffers *cb,
                                                Object *ob,
                                                const float *color)
{

    if(ob->rigidbody_object){
          if(ob->rigidbody_object->shared->col_shape_draw_data == NULL) {

          }
          else {
            GPUBatch *geom = DRW_cache_non_primitive_col_shape_get(ob);
            if(geom){
                OVERLAY_extra_wire(cb, geom, ob->obmat, color);
            }
          }
    }


}

static void OVERLAY_bounds(OVERLAY_ExtraCallBuffers *cb,
                           Object *ob,
                           const float *color,
                           char boundtype,
                           bool around_origin,
                           float mat[4][4])
{
  float center[3], size[3], tmp[4][4], final_mat[4][4];
  BoundBox bb_local;

  if (ob->type == OB_MBALL && !BKE_mball_is_basis(ob)) {
    return;
  }

  BoundBox *bb = BKE_object_boundbox_get(ob);

  if (bb == NULL) {
    const float min[3] = {-1.0f, -1.0f, -1.0f}, max[3] = {1.0f, 1.0f, 1.0f};
    bb = &bb_local;
    BKE_boundbox_init_from_minmax(bb, min, max);
  }

  BKE_boundbox_calc_size_aabb(bb, size);
  if(ob->parent) {
      if(ob->parent->rigidbody_object) {
          if(ob->parent->rigidbody_object->shape == RB_SHAPE_COMPOUND) {
              float transform_mat[4][4];
              float ob_size[3];
              float parent_size[3];
              mat4_to_size(ob_size, ob->obmat);
              mat4_to_size(parent_size, ob->parent->obmat);
              float transformed_obmat[4][4];
              copy_m4_m4(transformed_obmat, ob->parent->obmat);
              /* Remove arent scale factor and add object's scale. */
              for(int i=0; i<3; i++){
                  for(int j=0; j<3; j++){
                      transformed_obmat[i][j] = transformed_obmat[i][j] * (ob_size[j]/parent_size[j]);
                  }
              }
              copy_m4_m4(transform_mat, ob->parent->obmat);
              invert_m4(transform_mat);
              mul_m4_m4m4(transform_mat, transform_mat, transformed_obmat);
              /* This is the transform that is given to bullet while creating the collisions shapes,
               * so the extra scale factor that is applied to the collision shape
               * should be extracted from the same matrix. */
              invert_m4(transform_mat);

              mat4_to_size(parent_size, transform_mat);
              copy_v3_v3(size, parent_size);

          }
      }
  }

  if (around_origin) {
    zero_v3(center);
  }
  else {
    BKE_boundbox_calc_center_aabb(bb, center);
  }

  switch (boundtype) {
    case OB_BOUND_BOX:
      size_to_mat4(tmp, size);
      copy_v3_v3(tmp[3], center);
      mul_m4_m4m4(tmp, ob->obmat, tmp);
      DRW_buffer_add_entry(cb->empty_cube, color, tmp);
      break;
    case OB_BOUND_SPHERE:
      size[0] = max_fff(size[0], size[1], size[2]);
      size[1] = size[2] = size[0];
      size_to_mat4(tmp, size);
      copy_v3_v3(tmp[3], center);
      mul_m4_m4m4(tmp, ob->obmat, tmp);
      DRW_buffer_add_entry(cb->empty_sphere, color, tmp);
      break;
    case OB_BOUND_CYLINDER:
      size[0] = max_ff(size[0], size[1]);
      size[1] = size[0];
      size_to_mat4(tmp, size);
      copy_v3_v3(tmp[3], center);
      mul_m4_m4m4(tmp, ob->obmat, tmp);
      DRW_buffer_add_entry(cb->empty_cylinder, color, tmp);
      break;
    case OB_BOUND_CONE:
      size[0] = max_ff(size[0], size[1]);
      size[1] = size[0];
      size_to_mat4(tmp, size);
      copy_v3_v3(tmp[3], center);
      /* Cone batch has base at 0 and is pointing towards +Y. */
      swap_v3_v3(tmp[1], tmp[2]);
      tmp[3][2] -= size[2];
      mul_m4_m4m4(tmp, ob->obmat, tmp);
      DRW_buffer_add_entry(cb->empty_cone, color, tmp);
      break;
    case OB_BOUND_CAPSULE:
      size[0] = max_ff(size[0], size[1]);
      size[1] = size[0];
      scale_m4_fl(tmp, size[0]);
      copy_v2_v2(tmp[3], center);
      tmp[3][2] = center[2] + max_ff(0.0f, size[2] - size[0]);
      mul_m4_m4m4(final_mat, ob->obmat, tmp);
      DRW_buffer_add_entry(cb->empty_capsule_cap, color, final_mat);
      negate_v3(tmp[2]);
      tmp[3][2] = center[2] - max_ff(0.0f, size[2] - size[0]);
      mul_m4_m4m4(final_mat, ob->obmat, tmp);
      DRW_buffer_add_entry(cb->empty_capsule_cap, color, final_mat);
      tmp[2][2] = max_ff(0.0f, size[2] * 2.0f - size[0] * 2.0f);
      mul_m4_m4m4(final_mat, ob->obmat, tmp);
      DRW_buffer_add_entry(cb->empty_capsule_body, color, final_mat);
      break;
  }
  if (mat != NULL)
    copy_m4_m4(mat, tmp);
}

static void OVERLAY_collision(OVERLAY_ExtraCallBuffers *cb, Object *ob, const float *color)
{
  switch (ob->rigidbody_object->shape) {
    case RB_SHAPE_BOX:
      OVERLAY_bounds(cb, ob, color, OB_BOUND_BOX, true, NULL);
      break;
    case RB_SHAPE_SPHERE:
      OVERLAY_bounds(cb, ob, color, OB_BOUND_SPHERE, true, NULL);
      break;
    case RB_SHAPE_CONE:
      OVERLAY_bounds(cb, ob, color, OB_BOUND_CONE, true, NULL);
      break;
    case RB_SHAPE_CYLINDER:
      OVERLAY_bounds(cb, ob, color, OB_BOUND_CYLINDER, true, NULL);
      break;
    case RB_SHAPE_CAPSULE:
      OVERLAY_bounds(cb, ob, color, OB_BOUND_CAPSULE, true, NULL);
      break;
    case RB_SHAPE_CONVEXH:
      OVERLAY_non_primitive_collision_shape(cb, ob, color);
      break;
    case RB_SHAPE_TRIMESH:
      OVERLAY_non_primitive_collision_shape(cb, ob, color);
      break;
  }
}

static void OVERLAY_texture_space(OVERLAY_ExtraCallBuffers *cb, Object *ob, const float *color)
{
  if (ob->data == NULL) {
    return;
  }

  ID *ob_data = ob->data;
  float *texcoloc = NULL;
  float *texcosize = NULL;

  switch (GS(ob_data->name)) {
    case ID_ME:
      BKE_mesh_texspace_get_reference((Mesh *)ob_data, NULL, &texcoloc, &texcosize);
      break;
    case ID_CU: {
      Curve *cu = (Curve *)ob_data;
      BKE_curve_texspace_ensure(cu);
      texcoloc = cu->loc;
      texcosize = cu->size;
      break;
    }
    case ID_MB: {
      MetaBall *mb = (MetaBall *)ob_data;
      texcoloc = mb->loc;
      texcosize = mb->size;
      break;
    }
    case ID_HA:
    case ID_PT:
    case ID_VO: {
      /* No user defined texture space support. */
      break;
    }
    default:
      BLI_assert(0);
  }

  float mat[4][4];

  if (texcoloc != NULL && texcosize != NULL) {
    size_to_mat4(mat, texcosize);
    copy_v3_v3(mat[3], texcoloc);
  }
  else {
    unit_m4(mat);
  }

  mul_m4_m4m4(mat, ob->obmat, mat);

  DRW_buffer_add_entry(cb->empty_cube, color, mat);
}

static void OVERLAY_forcefield(OVERLAY_ExtraCallBuffers *cb, Object *ob, ViewLayer *view_layer)
{
  int theme_id = DRW_object_wire_theme_get(ob, view_layer, NULL);
  float *color = DRW_color_background_blend_get(theme_id);
  PartDeflect *pd = ob->pd;
  Curve *cu = (ob->type == OB_CURVE) ? ob->data : NULL;

  union {
    float mat[4][4];
    struct {
      float _pad00[3], size_x;
      float _pad01[3], size_y;
      float _pad02[3], size_z;
      float pos[3], _pad03[1];
    };
  } instdata;

  copy_m4_m4(instdata.mat, ob->obmat);
  instdata.size_x = instdata.size_y = instdata.size_z = ob->empty_drawsize;

  switch (pd->forcefield) {
    case PFIELD_FORCE:
      DRW_buffer_add_entry(cb->field_force, color, &instdata);
      break;
    case PFIELD_WIND:
      instdata.size_z = pd->f_strength;
      DRW_buffer_add_entry(cb->field_wind, color, &instdata);
      break;
    case PFIELD_VORTEX:
      instdata.size_y = (pd->f_strength < 0.0f) ? -instdata.size_y : instdata.size_y;
      DRW_buffer_add_entry(cb->field_vortex, color, &instdata);
      break;
    case PFIELD_GUIDE:
      if (cu && (cu->flag & CU_PATH) && ob->runtime.curve_cache->anim_path_accum_length) {
        instdata.size_x = instdata.size_y = instdata.size_z = pd->f_strength;
        float pos[4], tmp[3];
        BKE_where_on_path(ob, 0.0f, pos, tmp, NULL, NULL, NULL);
        copy_v3_v3(instdata.pos, ob->obmat[3]);
        translate_m4(instdata.mat, pos[0], pos[1], pos[2]);
        DRW_buffer_add_entry(cb->field_curve, color, &instdata);

        BKE_where_on_path(ob, 1.0f, pos, tmp, NULL, NULL, NULL);
        copy_v3_v3(instdata.pos, ob->obmat[3]);
        translate_m4(instdata.mat, pos[0], pos[1], pos[2]);
        DRW_buffer_add_entry(cb->field_sphere_limit, color, &instdata);
        /* Restore */
        copy_v3_v3(instdata.pos, ob->obmat[3]);
      }
      break;
  }

  if (pd->falloff == PFIELD_FALL_TUBE) {
    if (pd->flag & (PFIELD_USEMAX | PFIELD_USEMAXR)) {
      instdata.size_z = (pd->flag & PFIELD_USEMAX) ? pd->maxdist : 0.0f;
      instdata.size_x = (pd->flag & PFIELD_USEMAXR) ? pd->maxrad : 1.0f;
      instdata.size_y = instdata.size_x;
      DRW_buffer_add_entry(cb->field_tube_limit, color, &instdata);
    }
    if (pd->flag & (PFIELD_USEMIN | PFIELD_USEMINR)) {
      instdata.size_z = (pd->flag & PFIELD_USEMIN) ? pd->mindist : 0.0f;
      instdata.size_x = (pd->flag & PFIELD_USEMINR) ? pd->minrad : 1.0f;
      instdata.size_y = instdata.size_x;
      DRW_buffer_add_entry(cb->field_tube_limit, color, &instdata);
    }
  }
  else if (pd->falloff == PFIELD_FALL_CONE) {
    if (pd->flag & (PFIELD_USEMAX | PFIELD_USEMAXR)) {
      float radius = DEG2RADF((pd->flag & PFIELD_USEMAXR) ? pd->maxrad : 1.0f);
      float distance = (pd->flag & PFIELD_USEMAX) ? pd->maxdist : 0.0f;
      instdata.size_x = distance * sinf(radius);
      instdata.size_z = distance * cosf(radius);
      instdata.size_y = instdata.size_x;
      DRW_buffer_add_entry(cb->field_cone_limit, color, &instdata);
    }
    if (pd->flag & (PFIELD_USEMIN | PFIELD_USEMINR)) {
      float radius = DEG2RADF((pd->flag & PFIELD_USEMINR) ? pd->minrad : 1.0f);
      float distance = (pd->flag & PFIELD_USEMIN) ? pd->mindist : 0.0f;
      instdata.size_x = distance * sinf(radius);
      instdata.size_z = distance * cosf(radius);
      instdata.size_y = instdata.size_x;
      DRW_buffer_add_entry(cb->field_cone_limit, color, &instdata);
    }
  }
  else if (pd->falloff == PFIELD_FALL_SPHERE) {
    if (pd->flag & PFIELD_USEMAX) {
      instdata.size_x = instdata.size_y = instdata.size_z = pd->maxdist;
      DRW_buffer_add_entry(cb->field_sphere_limit, color, &instdata);
    }
    if (pd->flag & PFIELD_USEMIN) {
      instdata.size_x = instdata.size_y = instdata.size_z = pd->mindist;
      DRW_buffer_add_entry(cb->field_sphere_limit, color, &instdata);
    }
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Lights
 * \{ */

void OVERLAY_light_cache_populate(OVERLAY_Data *vedata, Object *ob)
{
  OVERLAY_ExtraCallBuffers *cb = OVERLAY_extra_call_buffer_get(vedata, ob);
  const DRWContextState *draw_ctx = DRW_context_state_get();
  ViewLayer *view_layer = draw_ctx->view_layer;

  Light *la = ob->data;
  float *color_p;
  DRW_object_wire_theme_get(ob, view_layer, &color_p);
  /* Remove the alpha. */
  float color[4] = {UNPACK3(color_p), 1.0f};
  /* Pack render data into object matrix. */
  union {
    float mat[4][4];
    struct {
      float _pad00[3];
      union {
        float area_size_x;
        float spot_cosine;
      };
      float _pad01[3];
      union {
        float area_size_y;
        float spot_blend;
      };
      float _pad02[3], clip_sta;
      float pos[3], clip_end;
    };
  } instdata;

  copy_m4_m4(instdata.mat, ob->obmat);
  /* FIXME / TODO: clip_end has no meaning nowadays.
   * In EEVEE, Only clip_sta is used shadow-mapping.
   * Clip end is computed automatically based on light power.
   * For now, always use the custom distance as clip_end. */
  instdata.clip_end = la->att_dist;
  instdata.clip_sta = la->clipsta;

  DRW_buffer_add_entry(cb->groundline, instdata.pos);

  if (la->type == LA_LOCAL) {
    instdata.area_size_x = instdata.area_size_y = la->area_size;
    DRW_buffer_add_entry(cb->light_point, color, &instdata);
  }
  else if (la->type == LA_SUN) {
    DRW_buffer_add_entry(cb->light_sun, color, &instdata);
  }
  else if (la->type == LA_SPOT) {
    /* Previous implementation was using the clipend distance as cone size.
     * We cannot do this anymore so we use a fixed size of 10. (see T72871) */
    rescale_m4(instdata.mat, (float[3]){10.0f, 10.0f, 10.0f});
    /* For cycles and eevee the spot attenuation is
     * y = (1/(1 + x^2) - a)/((1 - a) b)
     * We solve the case where spot attenuation y = 1 and y = 0
     * root for y = 1 is  (-1 - c) / c
     * root for y = 0 is  (1 - a) / a
     * and use that to position the blend circle. */
    float a = cosf(la->spotsize * 0.5f);
    float b = la->spotblend;
    float c = a * b - a - b;
    /* Optimized version or root1 / root0 */
    instdata.spot_blend = sqrtf((-a - c * a) / (c - c * a));
    instdata.spot_cosine = a;
    /* HACK: We pack the area size in alpha color. This is decoded by the shader. */
    color[3] = -max_ff(la->area_size, FLT_MIN);
    DRW_buffer_add_entry(cb->light_spot, color, &instdata);

    if ((la->mode & LA_SHOW_CONE) && !DRW_state_is_select()) {
      const float color_inside[4] = {0.0f, 0.0f, 0.0f, 0.5f};
      const float color_outside[4] = {1.0f, 1.0f, 1.0f, 0.3f};
      DRW_buffer_add_entry(cb->light_spot_cone_front, color_inside, &instdata);
      DRW_buffer_add_entry(cb->light_spot_cone_back, color_outside, &instdata);
    }
  }
  else if (la->type == LA_AREA) {
    bool uniform_scale = !ELEM(la->area_shape, LA_AREA_RECT, LA_AREA_ELLIPSE);
    int sqr = ELEM(la->area_shape, LA_AREA_SQUARE, LA_AREA_RECT);
    instdata.area_size_x = la->area_size;
    instdata.area_size_y = uniform_scale ? la->area_size : la->area_sizey;
    DRW_buffer_add_entry(cb->light_area[sqr], color, &instdata);
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Light-probe
 * \{ */

void OVERLAY_lightprobe_cache_populate(OVERLAY_Data *vedata, Object *ob)
{
  OVERLAY_ExtraCallBuffers *cb = OVERLAY_extra_call_buffer_get(vedata, ob);
  const DRWContextState *draw_ctx = DRW_context_state_get();
  ViewLayer *view_layer = draw_ctx->view_layer;
  float *color_p;
  int theme_id = DRW_object_wire_theme_get(ob, view_layer, &color_p);
  const LightProbe *prb = (LightProbe *)ob->data;
  const bool show_clipping = (prb->flag & LIGHTPROBE_FLAG_SHOW_CLIP_DIST) != 0;
  const bool show_parallax = (prb->flag & LIGHTPROBE_FLAG_SHOW_PARALLAX) != 0;
  const bool show_influence = (prb->flag & LIGHTPROBE_FLAG_SHOW_INFLUENCE) != 0;
  const bool show_data = (ob->base_flag & BASE_SELECTED) || DRW_state_is_select();

  union {
    float mat[4][4];
    struct {
      float _pad00[4];
      float _pad01[4];
      float _pad02[3], clip_sta;
      float pos[3], clip_end;
    };
  } instdata;

  copy_m4_m4(instdata.mat, ob->obmat);

  switch (prb->type) {
    case LIGHTPROBE_TYPE_CUBE:
      instdata.clip_sta = show_clipping ? prb->clipsta : -1.0;
      instdata.clip_end = show_clipping ? prb->clipend : -1.0;
      DRW_buffer_add_entry(cb->probe_cube, color_p, &instdata);
      DRW_buffer_add_entry(cb->groundline, instdata.pos);

      if (show_influence) {
        char shape = (prb->attenuation_type == LIGHTPROBE_SHAPE_BOX) ? OB_CUBE : OB_EMPTY_SPHERE;
        float f = 1.0f - prb->falloff;
        OVERLAY_empty_shape(cb, ob->obmat, prb->distinf, shape, color_p);
        OVERLAY_empty_shape(cb, ob->obmat, prb->distinf * f, shape, color_p);
      }

      if (show_parallax) {
        char shape = (prb->parallax_type == LIGHTPROBE_SHAPE_BOX) ? OB_CUBE : OB_EMPTY_SPHERE;
        float dist = ((prb->flag & LIGHTPROBE_FLAG_CUSTOM_PARALLAX) != 0) ? prb->distpar :
                                                                            prb->distinf;
        OVERLAY_empty_shape(cb, ob->obmat, dist, shape, color_p);
      }
      break;
    case LIGHTPROBE_TYPE_GRID:
      instdata.clip_sta = show_clipping ? prb->clipsta : -1.0;
      instdata.clip_end = show_clipping ? prb->clipend : -1.0;
      DRW_buffer_add_entry(cb->probe_grid, color_p, &instdata);

      if (show_influence) {
        float f = 1.0f - prb->falloff;
        OVERLAY_empty_shape(cb, ob->obmat, 1.0 + prb->distinf, OB_CUBE, color_p);
        OVERLAY_empty_shape(cb, ob->obmat, 1.0 + prb->distinf * f, OB_CUBE, color_p);
      }

      /* Data dots */
      if (show_data) {
        instdata.mat[0][3] = prb->grid_resolution_x;
        instdata.mat[1][3] = prb->grid_resolution_y;
        instdata.mat[2][3] = prb->grid_resolution_z;
        /* Put theme id in matrix. */
        if (theme_id == TH_ACTIVE) {
          instdata.mat[3][3] = 1.0;
        }
        else /* TH_SELECT */ {
          instdata.mat[3][3] = 2.0;
        }

        uint cell_count = prb->grid_resolution_x * prb->grid_resolution_y * prb->grid_resolution_z;
        DRWShadingGroup *grp = DRW_shgroup_create_sub(vedata->stl->pd->extra_grid_grp);
        DRW_shgroup_uniform_vec4_array_copy(grp, "gridModelMatrix", instdata.mat, 4);
        DRW_shgroup_call_procedural_points(grp, NULL, cell_count);
      }
      break;
    case LIGHTPROBE_TYPE_PLANAR:
      DRW_buffer_add_entry(cb->probe_planar, color_p, &instdata);

      if (DRW_state_is_select() && (prb->flag & LIGHTPROBE_FLAG_SHOW_DATA)) {
        DRW_buffer_add_entry(cb->solid_quad, color_p, &instdata);
      }

      if (show_influence) {
        normalize_v3_length(instdata.mat[2], prb->distinf);
        DRW_buffer_add_entry(cb->empty_cube, color_p, &instdata);
        mul_v3_fl(instdata.mat[2], 1.0f - prb->falloff);
        DRW_buffer_add_entry(cb->empty_cube, color_p, &instdata);
      }
      zero_v3(instdata.mat[2]);
      DRW_buffer_add_entry(cb->empty_cube, color_p, &instdata);

      normalize_m4_m4(instdata.mat, ob->obmat);
      OVERLAY_empty_shape(cb, instdata.mat, ob->empty_drawsize, OB_SINGLE_ARROW, color_p);
      break;
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Speaker
 * \{ */

void OVERLAY_speaker_cache_populate(OVERLAY_Data *vedata, Object *ob)
{
  OVERLAY_ExtraCallBuffers *cb = OVERLAY_extra_call_buffer_get(vedata, ob);
  const DRWContextState *draw_ctx = DRW_context_state_get();
  ViewLayer *view_layer = draw_ctx->view_layer;
  float *color_p;
  DRW_object_wire_theme_get(ob, view_layer, &color_p);

  DRW_buffer_add_entry(cb->speaker, color_p, ob->obmat);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Camera
 * \{ */

typedef union OVERLAY_CameraInstanceData {
  /* Pack render data into object matrix and object color. */
  struct {
    float color[4];
    float mat[4][4];
  };
  struct {
    float _pad0[2];
    float volume_sta;
    union {
      float depth;
      float focus;
      float volume_end;
    };
    float _pad00[3];
    union {
      float corner_x;
      float dist_color_id;
    };
    float _pad01[3];
    union {
      float corner_y;
    };
    float _pad02[3];
    union {
      float center_x;
      float clip_sta;
      float mist_sta;
    };
    float pos[3];
    union {
      float center_y;
      float clip_end;
      float mist_end;
    };
  };
} OVERLAY_CameraInstanceData;

static void camera_view3d_reconstruction(
    OVERLAY_ExtraCallBuffers *cb, Scene *scene, View3D *v3d, Object *ob, const float color[4])
{
  const DRWContextState *draw_ctx = DRW_context_state_get();
  const bool is_select = DRW_state_is_select();

  MovieClip *clip = BKE_object_movieclip_get(scene, ob, false);
  if (clip == NULL) {
    return;
  }

  const bool is_solid_bundle = (v3d->bundle_drawtype == OB_EMPTY_SPHERE) &&
                               ((v3d->shading.type != OB_SOLID) || !XRAY_FLAG_ENABLED(v3d));

  MovieTracking *tracking = &clip->tracking;
  /* Index must start in 1, to mimic BKE_tracking_track_get_indexed. */
  int track_index = 1;

  float bundle_color_custom[3];
  float *bundle_color_solid = G_draw.block.colorBundleSolid;
  float *bundle_color_unselected = G_draw.block.colorWire;
  uchar text_color_selected[4], text_color_unselected[4];
  /* Color Management: Exception here as texts are drawn in sRGB space directly. */
  UI_GetThemeColor4ubv(TH_SELECT, text_color_selected);
  UI_GetThemeColor4ubv(TH_TEXT, text_color_unselected);

  float camera_mat[4][4];
  BKE_tracking_get_camera_object_matrix(ob, camera_mat);

  LISTBASE_FOREACH (MovieTrackingObject *, tracking_object, &tracking->objects) {
    float tracking_object_mat[4][4];

    if (tracking_object->flag & TRACKING_OBJECT_CAMERA) {
      copy_m4_m4(tracking_object_mat, camera_mat);
    }
    else {
      const int framenr = BKE_movieclip_remap_scene_to_clip_frame(
          clip, DEG_get_ctime(draw_ctx->depsgraph));

      float object_mat[4][4];
      BKE_tracking_camera_get_reconstructed_interpolate(
          tracking, tracking_object, framenr, object_mat);

      float object_imat[4][4];
      invert_m4_m4(object_imat, object_mat);

      mul_m4_m4m4(tracking_object_mat, ob->obmat, object_imat);
    }

    ListBase *tracksbase = BKE_tracking_object_get_tracks(tracking, tracking_object);
    LISTBASE_FOREACH (MovieTrackingTrack *, track, tracksbase) {
      if ((track->flag & TRACK_HAS_BUNDLE) == 0) {
        continue;
      }
      bool is_selected = TRACK_SELECTED(track);

      float bundle_mat[4][4];
      copy_m4_m4(bundle_mat, tracking_object_mat);
      translate_m4(bundle_mat, track->bundle_pos[0], track->bundle_pos[1], track->bundle_pos[2]);

      const float *bundle_color;
      if (track->flag & TRACK_CUSTOMCOLOR) {
        /* Meh, hardcoded srgb transform here. */
        /* TODO: change the actual DNA color to be linear. */
        srgb_to_linearrgb_v3_v3(bundle_color_custom, track->color);
        bundle_color = bundle_color_custom;
      }
      else if (is_solid_bundle) {
        bundle_color = bundle_color_solid;
      }
      else if (is_selected) {
        bundle_color = color;
      }
      else {
        bundle_color = bundle_color_unselected;
      }

      if (is_select) {
        DRW_select_load_id(ob->runtime.select_id | (track_index << 16));
        track_index++;
      }

      if (is_solid_bundle) {
        if (is_selected) {
          OVERLAY_empty_shape(cb, bundle_mat, v3d->bundle_size, v3d->bundle_drawtype, color);
        }

        const float bundle_color_v4[4] = {
            bundle_color[0],
            bundle_color[1],
            bundle_color[2],
            1.0f,
        };

        bundle_mat[3][3] = v3d->bundle_size; /* See shader. */
        DRW_buffer_add_entry(cb->empty_sphere_solid, bundle_color_v4, bundle_mat);
      }
      else {
        OVERLAY_empty_shape(cb, bundle_mat, v3d->bundle_size, v3d->bundle_drawtype, bundle_color);
      }

      if ((v3d->flag2 & V3D_SHOW_BUNDLENAME) && !is_select) {
        struct DRWTextStore *dt = DRW_text_cache_ensure();

        DRW_text_cache_add(dt,
                           bundle_mat[3],
                           track->name,
                           strlen(track->name),
                           10,
                           0,
                           DRW_TEXT_CACHE_GLOBALSPACE | DRW_TEXT_CACHE_STRING_PTR,
                           is_selected ? text_color_selected : text_color_unselected);
      }
    }

    if ((v3d->flag2 & V3D_SHOW_CAMERAPATH) && (tracking_object->flag & TRACKING_OBJECT_CAMERA) &&
        !is_select) {
      MovieTrackingReconstruction *reconstruction;
      reconstruction = BKE_tracking_object_get_reconstruction(tracking, tracking_object);

      if (reconstruction->camnr) {
        MovieReconstructedCamera *camera = reconstruction->cameras;
        float v0[3], v1[3];
        for (int a = 0; a < reconstruction->camnr; a++, camera++) {
          copy_v3_v3(v0, v1);
          copy_v3_v3(v1, camera->mat[3]);
          mul_m4_v3(camera_mat, v1);
          if (a > 0) {
            /* This one is suboptimal (gl_lines instead of gl_line_strip)
             * but we keep this for simplicity */
            OVERLAY_extra_line(cb, v0, v1, TH_CAMERA_PATH);
          }
        }
      }
    }
  }
}

static float camera_offaxis_shiftx_get(Scene *scene,
                                       Object *ob,
                                       const OVERLAY_CameraInstanceData *instdata,
                                       bool right_eye)
{
  Camera *cam = ob->data;
  if (cam->stereo.convergence_mode == CAM_S3D_OFFAXIS) {
    const char *viewnames[2] = {STEREO_LEFT_NAME, STEREO_RIGHT_NAME};
    const float shiftx = BKE_camera_multiview_shift_x(&scene->r, ob, viewnames[right_eye]);
    const float delta_shiftx = shiftx - cam->shiftx;
    const float width = instdata->corner_x * 2.0f;
    return delta_shiftx * width;
  }

  return 0.0;
}
/**
 * Draw the stereo 3d support elements (cameras, plane, volume).
 * They are only visible when not looking through the camera:
 */
static void camera_stereoscopy_extra(OVERLAY_ExtraCallBuffers *cb,
                                     Scene *scene,
                                     View3D *v3d,
                                     Object *ob,
                                     const OVERLAY_CameraInstanceData *instdata)
{
  OVERLAY_CameraInstanceData stereodata = *instdata;
  Camera *cam = ob->data;
  const bool is_select = DRW_state_is_select();
  const char *viewnames[2] = {STEREO_LEFT_NAME, STEREO_RIGHT_NAME};

  const bool is_stereo3d_cameras = (v3d->stereo3d_flag & V3D_S3D_DISPCAMERAS) != 0;
  const bool is_stereo3d_plane = (v3d->stereo3d_flag & V3D_S3D_DISPPLANE) != 0;
  const bool is_stereo3d_volume = (v3d->stereo3d_flag & V3D_S3D_DISPVOLUME) != 0;

  if (!is_stereo3d_cameras) {
    /* Draw single camera. */
    DRW_buffer_add_entry_struct(cb->camera_frame, instdata);
  }

  for (int eye = 0; eye < 2; eye++) {
    ob = BKE_camera_multiview_render(scene, ob, viewnames[eye]);
    BKE_camera_multiview_model_matrix(&scene->r, ob, viewnames[eye], stereodata.mat);

    stereodata.corner_x = instdata->corner_x;
    stereodata.corner_y = instdata->corner_y;
    stereodata.center_x = instdata->center_x + camera_offaxis_shiftx_get(scene, ob, instdata, eye);
    stereodata.center_y = instdata->center_y;
    stereodata.depth = instdata->depth;

    if (is_stereo3d_cameras) {
      DRW_buffer_add_entry_struct(cb->camera_frame, &stereodata);

      /* Connecting line between cameras. */
      OVERLAY_extra_line_dashed(cb, stereodata.pos, instdata->pos, G_draw.block.colorWire);
    }

    if (is_stereo3d_volume && !is_select) {
      float r = (eye == 1) ? 2.0f : 1.0f;

      stereodata.volume_sta = -cam->clip_start;
      stereodata.volume_end = -cam->clip_end;
      /* Encode eye + intensity and alpha (see shader) */
      copy_v2_fl2(stereodata.color, r + 0.15f, 1.0f);
      DRW_buffer_add_entry_struct(cb->camera_volume_frame, &stereodata);

      if (v3d->stereo3d_volume_alpha > 0.0f) {
        /* Encode eye + intensity and alpha (see shader) */
        copy_v2_fl2(stereodata.color, r + 0.999f, v3d->stereo3d_volume_alpha);
        DRW_buffer_add_entry_struct(cb->camera_volume, &stereodata);
      }
      /* restore */
      copy_v3_v3(stereodata.color, instdata->color);
    }
  }

  if (is_stereo3d_plane && !is_select) {
    if (cam->stereo.convergence_mode == CAM_S3D_TOE) {
      /* There is no real convergence plane but we highlight the center
       * point where the views are pointing at. */
      // zero_v3(stereodata.mat[0]); /* We reconstruct from Z and Y */
      // zero_v3(stereodata.mat[1]); /* Y doesn't change */
      zero_v3(stereodata.mat[2]);
      zero_v3(stereodata.mat[3]);
      for (int i = 0; i < 2; i++) {
        float mat[4][4];
        /* Need normalized version here. */
        BKE_camera_multiview_model_matrix(&scene->r, ob, viewnames[i], mat);
        add_v3_v3(stereodata.mat[2], mat[2]);
        madd_v3_v3fl(stereodata.mat[3], mat[3], 0.5f);
      }
      normalize_v3(stereodata.mat[2]);
      cross_v3_v3v3(stereodata.mat[0], stereodata.mat[1], stereodata.mat[2]);
    }
    else if (cam->stereo.convergence_mode == CAM_S3D_PARALLEL) {
      /* Show plane at the given distance between the views even if it makes no sense. */
      zero_v3(stereodata.pos);
      for (int i = 0; i < 2; i++) {
        float mat[4][4];
        BKE_camera_multiview_model_matrix_scaled(&scene->r, ob, viewnames[i], mat);
        madd_v3_v3fl(stereodata.pos, mat[3], 0.5f);
      }
    }
    else if (cam->stereo.convergence_mode == CAM_S3D_OFFAXIS) {
      /* Nothing to do. Everything is already setup. */
    }
    stereodata.volume_sta = -cam->stereo.convergence_distance;
    stereodata.volume_end = -cam->stereo.convergence_distance;
    /* Encode eye + intensity and alpha (see shader) */
    copy_v2_fl2(stereodata.color, 0.1f, 1.0f);
    DRW_buffer_add_entry_struct(cb->camera_volume_frame, &stereodata);

    if (v3d->stereo3d_convergence_alpha > 0.0f) {
      /* Encode eye + intensity and alpha (see shader) */
      copy_v2_fl2(stereodata.color, 0.0f, v3d->stereo3d_convergence_alpha);
      DRW_buffer_add_entry_struct(cb->camera_volume, &stereodata);
    }
  }
}

void OVERLAY_camera_cache_populate(OVERLAY_Data *vedata, Object *ob)
{
  OVERLAY_ExtraCallBuffers *cb = OVERLAY_extra_call_buffer_get(vedata, ob);
  OVERLAY_CameraInstanceData instdata;

  const DRWContextState *draw_ctx = DRW_context_state_get();
  ViewLayer *view_layer = draw_ctx->view_layer;
  View3D *v3d = draw_ctx->v3d;
  Scene *scene = draw_ctx->scene;
  RegionView3D *rv3d = draw_ctx->rv3d;

  Camera *cam = ob->data;
  Object *camera_object = DEG_get_evaluated_object(draw_ctx->depsgraph, v3d->camera);
  const bool is_select = DRW_state_is_select();
  const bool is_active = (ob == camera_object);
  const bool look_through = (is_active && (rv3d->persp == RV3D_CAMOB));

  const bool is_multiview = (scene->r.scemode & R_MULTIVIEW) != 0;
  const bool is_stereo3d_view = (scene->r.views_format == SCE_VIEWS_FORMAT_STEREO_3D);
  const bool is_stereo3d_display_extra = is_active && is_multiview && (!look_through) &&
                                         ((v3d->stereo3d_flag) != 0);
  const bool is_selection_camera_stereo = is_select && look_through && is_multiview &&
                                          is_stereo3d_view;

  float vec[4][3], asp[2], shift[2], scale[3], drawsize, center[2], corner[2];

  float *color_p;
  DRW_object_wire_theme_get(ob, view_layer, &color_p);
  copy_v4_v4(instdata.color, color_p);

  normalize_m4_m4(instdata.mat, ob->obmat);

  /* BKE_camera_multiview_model_matrix already accounts for scale, don't do it here. */
  if (is_selection_camera_stereo) {
    copy_v3_fl(scale, 1.0f);
  }
  else {
    copy_v3_fl3(scale, len_v3(ob->obmat[0]), len_v3(ob->obmat[1]), len_v3(ob->obmat[2]));
    /* Avoid division by 0. */
    if (ELEM(0.0f, scale[0], scale[1], scale[2])) {
      return;
    }
    invert_v3(scale);
  }

  BKE_camera_view_frame_ex(
      scene, cam, cam->drawsize, look_through, scale, asp, shift, &drawsize, vec);

  /* Apply scale to simplify the rest of the drawing. */
  invert_v3(scale);
  for (int i = 0; i < 4; i++) {
    mul_v3_v3(vec[i], scale);
    /* Project to z=-1 plane. Makes positioning / scaling easier. (see shader) */
    mul_v2_fl(vec[i], 1.0f / fabsf(vec[i][2]));
  }

  /* Frame coords */
  mid_v2_v2v2(center, vec[0], vec[2]);
  sub_v2_v2v2(corner, vec[0], center);
  instdata.corner_x = corner[0];
  instdata.corner_y = corner[1];
  instdata.center_x = center[0];
  instdata.center_y = center[1];
  instdata.depth = vec[0][2];

  if (look_through) {
    if (!DRW_state_is_image_render()) {
      /* Only draw the frame. */
      if (is_multiview) {
        float mat[4][4];
        const bool is_right = v3d->multiview_eye == STEREO_RIGHT_ID;
        const char *view_name = is_right ? STEREO_RIGHT_NAME : STEREO_LEFT_NAME;
        BKE_camera_multiview_model_matrix(&scene->r, ob, view_name, mat);
        instdata.center_x += camera_offaxis_shiftx_get(scene, ob, &instdata, is_right);
        for (int i = 0; i < 4; i++) {
          /* Partial copy to avoid overriding packed data. */
          copy_v3_v3(instdata.mat[i], mat[i]);
        }
      }
      instdata.depth = -instdata.depth; /* Hides the back of the camera wires (see shader). */
      DRW_buffer_add_entry_struct(cb->camera_frame, &instdata);
    }
  }
  else {
    /* Stereo cameras, volumes, plane drawing. */
    if (is_stereo3d_display_extra) {
      camera_stereoscopy_extra(cb, scene, v3d, ob, &instdata);
    }
    else {
      DRW_buffer_add_entry_struct(cb->camera_frame, &instdata);
    }
  }

  if (!look_through) {
    /* Triangle. */
    float tria_size = 0.7f * drawsize / fabsf(instdata.depth);
    float tria_margin = 0.1f * drawsize / fabsf(instdata.depth);
    instdata.center_x = center[0];
    instdata.center_y = center[1] + instdata.corner_y + tria_margin + tria_size;
    instdata.corner_x = instdata.corner_y = -tria_size;
    DRW_buffer_add_entry_struct(cb->camera_tria[is_active], &instdata);
  }

  if (cam->flag & CAM_SHOWLIMITS) {
    /* Scale focus point. */
    mul_v3_fl(instdata.mat[0], cam->drawsize);
    mul_v3_fl(instdata.mat[1], cam->drawsize);

    instdata.dist_color_id = (is_active) ? 3 : 2;
    instdata.focus = -BKE_camera_object_dof_distance(ob);
    instdata.clip_sta = cam->clip_start;
    instdata.clip_end = cam->clip_end;
    DRW_buffer_add_entry_struct(cb->camera_distances, &instdata);
  }

  if (cam->flag & CAM_SHOWMIST) {
    World *world = scene->world;
    if (world) {
      instdata.dist_color_id = (is_active) ? 1 : 0;
      instdata.focus = 1.0f; /* Disable */
      instdata.mist_sta = world->miststa;
      instdata.mist_end = world->miststa + world->mistdist;
      DRW_buffer_add_entry_struct(cb->camera_distances, &instdata);
    }
  }

  /* Motion Tracking. */
  if ((v3d->flag2 & V3D_SHOW_RECONSTRUCTION) != 0) {
    camera_view3d_reconstruction(cb, scene, v3d, ob, color_p);
  }

  /* Background images. */
  if (look_through && (cam->flag & CAM_SHOW_BG_IMAGE) && !BLI_listbase_is_empty(&cam->bg_images)) {
    OVERLAY_image_camera_cache_populate(vedata, ob);
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Relationships & constraints
 * \{ */

static void OVERLAY_relationship_lines(OVERLAY_ExtraCallBuffers *cb,
                                       Depsgraph *depsgraph,
                                       Scene *scene,
                                       Object *ob)
{
  float *relation_color = G_draw.block.colorWire;
  float *constraint_color = G_draw.block.colorGridAxisZ; /* ? */

  if (ob->parent && (DRW_object_visibility_in_active_context(ob->parent) & OB_VISIBLE_SELF)) {
    float *parent_pos = ob->runtime.parent_display_origin;
    OVERLAY_extra_line_dashed(cb, parent_pos, ob->obmat[3], relation_color);
  }

  /* Drawing the hook lines. */
  for (ModifierData *md = ob->modifiers.first; md; md = md->next) {
    if (md->type == eModifierType_Hook) {
      HookModifierData *hmd = (HookModifierData *)md;
      float center[3];
      mul_v3_m4v3(center, ob->obmat, hmd->cent);
      if (hmd->object) {
        OVERLAY_extra_line_dashed(cb, hmd->object->obmat[3], center, relation_color);
      }
      OVERLAY_extra_point(cb, center, relation_color);
    }
  }

  if (ob->rigidbody_constraint) {
    Object *rbc_ob1 = ob->rigidbody_constraint->ob1;
    Object *rbc_ob2 = ob->rigidbody_constraint->ob2;
    if (rbc_ob1 && (DRW_object_visibility_in_active_context(rbc_ob1) & OB_VISIBLE_SELF)) {
      OVERLAY_extra_line_dashed(cb, rbc_ob1->obmat[3], ob->obmat[3], relation_color);
    }
    if (rbc_ob2 && (DRW_object_visibility_in_active_context(rbc_ob2) & OB_VISIBLE_SELF)) {
      OVERLAY_extra_line_dashed(cb, rbc_ob2->obmat[3], ob->obmat[3], relation_color);
    }
  }

  /* Drawing the constraint lines */
  if (!BLI_listbase_is_empty(&ob->constraints)) {
    bConstraint *curcon;
    bConstraintOb *cob;
    ListBase *list = &ob->constraints;

    cob = BKE_constraints_make_evalob(depsgraph, scene, ob, NULL, CONSTRAINT_OBTYPE_OBJECT);

    for (curcon = list->first; curcon; curcon = curcon->next) {
      if (ELEM(curcon->type, CONSTRAINT_TYPE_FOLLOWTRACK, CONSTRAINT_TYPE_OBJECTSOLVER)) {
        /* special case for object solver and follow track constraints because they don't fill
         * constraint targets properly (design limitation -- scene is needed for their target
         * but it can't be accessed from get_targets callback) */
        Object *camob = NULL;

        if (curcon->type == CONSTRAINT_TYPE_FOLLOWTRACK) {
          bFollowTrackConstraint *data = (bFollowTrackConstraint *)curcon->data;
          camob = data->camera ? data->camera : scene->camera;
        }
        else if (curcon->type == CONSTRAINT_TYPE_OBJECTSOLVER) {
          bObjectSolverConstraint *data = (bObjectSolverConstraint *)curcon->data;
          camob = data->camera ? data->camera : scene->camera;
        }

        if (camob) {
          OVERLAY_extra_line_dashed(cb, camob->obmat[3], ob->obmat[3], constraint_color);
        }
      }
      else {
        const bConstraintTypeInfo *cti = BKE_constraint_typeinfo_get(curcon);

        if ((cti && cti->get_constraint_targets) && (curcon->ui_expand_flag & (1 << 0))) {
          ListBase targets = {NULL, NULL};
          bConstraintTarget *ct;

          cti->get_constraint_targets(curcon, &targets);

          for (ct = targets.first; ct; ct = ct->next) {
            /* calculate target's matrix */
            if (cti->get_target_matrix) {
              cti->get_target_matrix(depsgraph, curcon, cob, ct, DEG_get_ctime(depsgraph));
            }
            else {
              unit_m4(ct->matrix);
            }
            OVERLAY_extra_line_dashed(cb, ct->matrix[3], ob->obmat[3], constraint_color);
          }

          if (cti->flush_constraint_targets) {
            cti->flush_constraint_targets(curcon, &targets, 1);
          }
        }
      }
    }
    /* NOTE: Don't use BKE_constraints_clear_evalob here as that will reset ob->constinv. */
    MEM_freeN(cob);
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Volumetric / Smoke sim
 * \{ */

static void OVERLAY_volume_extra(OVERLAY_ExtraCallBuffers *cb,
                                 OVERLAY_Data *data,
                                 Object *ob,
                                 ModifierData *md,
                                 Scene *scene,
                                 const float *color)
{
  FluidModifierData *fmd = (FluidModifierData *)md;
  FluidDomainSettings *fds = fmd->domain;

  /* Don't show smoke before simulation starts, this could be made an option in the future. */
  const bool draw_velocity = (fds->draw_velocity && fds->fluid &&
                              CFRA >= fds->point_cache[0]->startframe);

  /* Show gridlines only for slices with no interpolation. */
  const bool show_gridlines = (fds->show_gridlines && fds->fluid &&
                               fds->axis_slice_method == AXIS_SLICE_SINGLE &&
                               (fds->interp_method == FLUID_DISPLAY_INTERP_CLOSEST ||
                                fds->coba_field == FLUID_DOMAIN_FIELD_FLAGS));

  const bool color_with_flags = (fds->gridlines_color_field == FLUID_GRIDLINE_COLOR_TYPE_FLAGS);

  const bool color_range = (fds->gridlines_color_field == FLUID_GRIDLINE_COLOR_TYPE_RANGE &&
                            fds->use_coba && fds->coba_field != FLUID_DOMAIN_FIELD_FLAGS);

  /* Small cube showing voxel size. */
  {
    float min[3];
    madd_v3fl_v3fl_v3fl_v3i(min, fds->p0, fds->cell_size, fds->res_min);
    float voxel_cubemat[4][4] = {{0.0f}};
    /* scale small cube to voxel size */
    voxel_cubemat[0][0] = fds->cell_size[0] / 2.0f;
    voxel_cubemat[1][1] = fds->cell_size[1] / 2.0f;
    voxel_cubemat[2][2] = fds->cell_size[2] / 2.0f;
    voxel_cubemat[3][3] = 1.0f;
    /* translate small cube to corner */
    copy_v3_v3(voxel_cubemat[3], min);
    /* move small cube into the domain (otherwise its centered on vertex of domain object) */
    translate_m4(voxel_cubemat, 1.0f, 1.0f, 1.0f);
    mul_m4_m4m4(voxel_cubemat, ob->obmat, voxel_cubemat);

    DRW_buffer_add_entry(cb->empty_cube, color, voxel_cubemat);
  }

  int slice_axis = -1;

  if (fds->axis_slice_method == AXIS_SLICE_SINGLE) {
    float viewinv[4][4];
    DRW_view_viewmat_get(NULL, viewinv, true);

    const int axis = (fds->slice_axis == SLICE_AXIS_AUTO) ? axis_dominant_v3_single(viewinv[2]) :
                                                            fds->slice_axis - 1;
    slice_axis = axis;
  }

  if (draw_velocity) {
    const bool use_needle = (fds->vector_draw_type == VECTOR_DRAW_NEEDLE);
    const bool use_mac = (fds->vector_draw_type == VECTOR_DRAW_MAC);
    const bool draw_mac_x = (fds->vector_draw_mac_components & VECTOR_DRAW_MAC_X);
    const bool draw_mac_y = (fds->vector_draw_mac_components & VECTOR_DRAW_MAC_Y);
    const bool draw_mac_z = (fds->vector_draw_mac_components & VECTOR_DRAW_MAC_Z);
    const bool cell_centered = (fds->vector_field == FLUID_DOMAIN_VECTOR_FIELD_FORCE);
    int line_count = 1;
    if (use_needle) {
      line_count = 6;
    }
    else if (use_mac) {
      line_count = 3;
    }
    line_count *= fds->res[0] * fds->res[1] * fds->res[2];

    if (fds->axis_slice_method == AXIS_SLICE_SINGLE) {
      line_count /= fds->res[slice_axis];
    }

    DRW_smoke_ensure_velocity(fmd);

    GPUShader *sh = OVERLAY_shader_volume_velocity(use_needle, use_mac);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, data->psl->extra_ps[0]);
    DRW_shgroup_uniform_texture(grp, "velocityX", fds->tex_velocity_x);
    DRW_shgroup_uniform_texture(grp, "velocityY", fds->tex_velocity_y);
    DRW_shgroup_uniform_texture(grp, "velocityZ", fds->tex_velocity_z);
    DRW_shgroup_uniform_float_copy(grp, "displaySize", fds->vector_scale);
    DRW_shgroup_uniform_float_copy(grp, "slicePosition", fds->slice_depth);
    DRW_shgroup_uniform_vec3_copy(grp, "cellSize", fds->cell_size);
    DRW_shgroup_uniform_vec3_copy(grp, "domainOriginOffset", fds->p0);
    DRW_shgroup_uniform_ivec3_copy(grp, "adaptiveCellOffset", fds->res_min);
    DRW_shgroup_uniform_int_copy(grp, "sliceAxis", slice_axis);
    DRW_shgroup_uniform_bool_copy(grp, "scaleWithMagnitude", fds->vector_scale_with_magnitude);
    DRW_shgroup_uniform_bool_copy(grp, "isCellCentered", cell_centered);

    if (use_mac) {
      DRW_shgroup_uniform_bool_copy(grp, "drawMACX", draw_mac_x);
      DRW_shgroup_uniform_bool_copy(grp, "drawMACY", draw_mac_y);
      DRW_shgroup_uniform_bool_copy(grp, "drawMACZ", draw_mac_z);
    }

    DRW_shgroup_call_procedural_lines(grp, ob, line_count);
  }

  if (show_gridlines) {
    GPUShader *sh = OVERLAY_shader_volume_gridlines(color_with_flags, color_range);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, data->psl->extra_ps[0]);
    DRW_shgroup_uniform_ivec3_copy(grp, "volumeSize", fds->res);
    DRW_shgroup_uniform_float_copy(grp, "slicePosition", fds->slice_depth);
    DRW_shgroup_uniform_vec3_copy(grp, "cellSize", fds->cell_size);
    DRW_shgroup_uniform_vec3_copy(grp, "domainOriginOffset", fds->p0);
    DRW_shgroup_uniform_ivec3_copy(grp, "adaptiveCellOffset", fds->res_min);
    DRW_shgroup_uniform_int_copy(grp, "sliceAxis", slice_axis);

    if (color_with_flags || color_range) {
      DRW_fluid_ensure_flags(fmd);
      DRW_shgroup_uniform_texture(grp, "flagTexture", fds->tex_flags);
    }

    if (color_range) {
      DRW_fluid_ensure_range_field(fmd);
      DRW_shgroup_uniform_texture(grp, "fieldTexture", fds->tex_range_field);
      DRW_shgroup_uniform_float_copy(grp, "lowerBound", fds->gridlines_lower_bound);
      DRW_shgroup_uniform_float_copy(grp, "upperBound", fds->gridlines_upper_bound);
      DRW_shgroup_uniform_vec4_copy(grp, "rangeColor", fds->gridlines_range_color);
      DRW_shgroup_uniform_int_copy(grp, "cellFilter", fds->gridlines_cell_filter);
    }

    const int line_count = 4 * fds->res[0] * fds->res[1] * fds->res[2] / fds->res[slice_axis];
    DRW_shgroup_call_procedural_lines(grp, ob, line_count);
  }

  if (draw_velocity || show_gridlines) {
    BLI_addtail(&data->stl->pd->smoke_domains, BLI_genericNodeN(fmd));
  }
}

static void OVERLAY_volume_free_smoke_textures(OVERLAY_Data *data)
{
  /* Free Smoke Textures after rendering */
  /* XXX This is a waste of processing and GPU bandwidth if nothing
   * is updated. But the problem is since Textures are stored in the
   * modifier we don't want them to take precious VRAM if the
   * modifier is not used for display. We should share them for
   * all viewport in a redraw at least. */
  LinkData *link;
  while ((link = BLI_pophead(&data->stl->pd->smoke_domains))) {
    FluidModifierData *fmd = (FluidModifierData *)link->data;
    DRW_smoke_free_velocity(fmd);
    MEM_freeN(link);
  }
}

/** \} */

/* -------------------------------------------------------------------- */

static void OVERLAY_object_center(OVERLAY_ExtraCallBuffers *cb,
                                  Object *ob,
                                  OVERLAY_PrivateData *pd,
                                  ViewLayer *view_layer)
{
  const bool is_library = ID_REAL_USERS(&ob->id) > 1 || ID_IS_LINKED(ob);

  if (ob == OBACT(view_layer)) {
    DRW_buffer_add_entry(cb->center_active, ob->obmat[3]);
  }
  else if (ob->base_flag & BASE_SELECTED) {
    DRWCallBuffer *cbuf = (is_library) ? cb->center_selected_lib : cb->center_selected;
    DRW_buffer_add_entry(cbuf, ob->obmat[3]);
  }
  else if (pd->v3d_flag & V3D_DRAW_CENTERS) {
    DRWCallBuffer *cbuf = (is_library) ? cb->center_deselected_lib : cb->center_deselected;
    DRW_buffer_add_entry(cbuf, ob->obmat[3]);
  }
}

static void OVERLAY_object_name(Object *ob, int theme_id)
{
  struct DRWTextStore *dt = DRW_text_cache_ensure();
  uchar color[4];
  /* Color Management: Exception here as texts are drawn in sRGB space directly. */
  UI_GetThemeColor4ubv(theme_id, color);

  DRW_text_cache_add(dt,
                     ob->obmat[3],
                     ob->id.name + 2,
                     strlen(ob->id.name + 2),
                     10,
                     0,
                     DRW_TEXT_CACHE_GLOBALSPACE | DRW_TEXT_CACHE_STRING_PTR,
                     color);
}

static void scale_vec_by_magnitude(float vector[3], float min_clamp, float scale, float pos[3])
{
  /* Vectors as scaled by multiplying by Scale and then adding "min_clamp"
   * Vectors are only drawn if they are greater than 0
   * Otherwise zero vectors will be drawn with magnitude = min_clamp". */
  float min_clamp_vec[3];
  float vec_len = len_v3(vector);
  normalize_v3(vector);
  copy_v3_v3(min_clamp_vec, vector);
  mul_v3_fl(min_clamp_vec, (min_clamp + 0.2f));
  mul_v3_fl(vector, scale * vec_len);
  add_v3_v3(vector, min_clamp_vec);
  add_v3_v3(vector, pos);
}

static void OVERLAY_vector_extra(OVERLAY_Data *data,
                                 float vector[3],
                                 float pos[3],
                                 float scale,
                                 float min_clamp,
                                 float color[3],
                                 int draw_text)
{

  uchar text_color[4];
  UI_GetThemeColor4ubv(TH_DRAWEXTRA_EDGELEN, text_color);

  float vector_head_pos[3];
  char vec_magnitude[9];
  copy_v3_v3(vector_head_pos, vector);
  float vec_len = len_v3(vector);
  BLI_snprintf(vec_magnitude, 9, "%f", vec_len);
  /* Scale the vector */
  scale_vec_by_magnitude(vector_head_pos, min_clamp, scale, pos);

  GPUShader *sh = OVERLAY_shader_vector();
  DRWShadingGroup *grp = DRW_shgroup_create(sh, data->psl->extra_ps[1]);
  DRW_shgroup_uniform_vec3_copy(grp, "objPosition", pos);
  DRW_shgroup_uniform_vec3_copy(grp, "vector", vector);
  DRW_shgroup_uniform_float_copy(grp, "scale", scale);
  DRW_shgroup_uniform_float_copy(grp, "min_clamp", min_clamp);
  DRW_shgroup_uniform_vec3_copy(grp, "colour", color);
  DRW_shgroup_call_procedural_lines(grp, NULL, 3);

  /* Draw magnitude of vector as text */
  if (draw_text) {
    struct DRWTextStore *dt = DRW_text_cache_ensure();
    DRW_text_cache_add(dt,
                       vector_head_pos,
                       vec_magnitude,
                       strlen(vec_magnitude),
                       -25,
                       0,
                       DRW_TEXT_CACHE_GLOBALSPACE,
                       text_color);
  }
}

static void OVERLAY_forces_extra(OVERLAY_Data *data, Scene *scene, RigidBodyOb *rbo)
{

  float scale = 0.05f;
  float min_clamp = 1.0f;
  float vector[3] = {0.0f};
  float color1[3] = {1.0, 0.0, 1.0}; /* Pink. */
  float color2[3] = {0.0, 1.0, 0.5}; /* Cyan. */
  float color3[3] = {1.0, 1.0, 0.0}; /* Yellow. */

  int draw_text = rbo->sim_display_options & RB_SIM_TEXT;

  if (rbo->display_force_types & RB_SIM_NET_FORCE) {
    /* Calculate net force. */
    float net_force[3];

    copy_v3_v3(net_force, scene->physics_settings.gravity);
    mul_v3_fl(net_force, rbo->mass);
    for (int i = 0; i < 3; i++) {
      add_v3_v3(net_force, rbo->eff_forces[i].vector);
      add_v3_v3(net_force, rbo->norm_forces[i].vector);
      add_v3_v3(net_force, rbo->fric_forces[i].vector);
    }

    if ((len_v3(net_force) > 0.000001f)) {
      OVERLAY_vector_extra(data, net_force, rbo->pos, scale, min_clamp, color2, draw_text);
    }
  }

  if (rbo->display_force_types & RB_SIM_GRAVITY) {
    /* Draw the force of gravity. */

    copy_v3_v3(vector, scene->physics_settings.gravity);
    mul_v3_fl(vector, rbo->mass);
    OVERLAY_vector_extra(data, vector, rbo->pos, scale, min_clamp, color1, draw_text);
  }

  if (rbo->display_force_types & RB_SIM_EFFECTORS) {

    for (int i = 0; i < 3; i++) {
      if (len_v3(rbo->eff_forces[i].vector)>0.000001f) {
        OVERLAY_vector_extra(
            data, rbo->eff_forces[i].vector, rbo->pos, scale, min_clamp, color1, draw_text);
      }
    }
  }

  if (rbo->display_force_types & RB_SIM_NORMAL) {
    for (int i = 0; i < 3; i++) {
      if (len_v3(rbo->norm_forces[i].vector)>0.000001f) {
        OVERLAY_vector_extra(data,
                             rbo->norm_forces[i].vector,
                             rbo->vec_locations[i].vector,
                             scale,
                             min_clamp,
                             color1,
                             draw_text);
      }
    }
  }
  if (rbo->display_force_types & RB_SIM_FRICTION) {
    for (int i = 0; i < 3; i++) {
      if (len_v3(rbo->fric_forces[i].vector)>0.000001f) {
        OVERLAY_vector_extra(data,
                             rbo->fric_forces[i].vector,
                             rbo->vec_locations[i].vector,
                             scale,
                             min_clamp,
                             color3,
                             draw_text);
      }
    }
  }
}

#ifdef WITH_BULLET
static void OVERLAY_velocity_extra(OVERLAY_Data *data, RigidBodyOb *rbo)
{
  float scale = 0.5f;
  float min_clamp = 1.0f;
  float color[3] = {1.0, 0.5, 1.0};

  int text_flag = rbo->sim_display_options & RB_SIM_TEXT;

  float vel[3] = {0.0f};
    copy_v3_v3(vel, rbo->vel);
  OVERLAY_vector_extra(data, vel, rbo->pos, scale, min_clamp, color, text_flag);
}

static void OVERLAY_acceleration_extra(OVERLAY_Data *data,
                                       RigidBodyOb *rbo,
                                       Depsgraph *depsgraph,
                                       Scene *scene)
{
  float scale = 0.5f;
  float min_clamp = 1.0f;
  float color[3] = {1.0, 1.0, 0.0};

  int text_flag = rbo->sim_display_options & RB_SIM_TEXT;

  /* Calculate timestep. */
  const float timestep = (1.0f / (float)FPS * scene->rigidbody_world->time_scale);

  float acc[3];
  sub_v3_v3v3(acc, rbo->vel, rbo->pvel);
  mul_v3_fl(acc, 1 / timestep);
  OVERLAY_vector_extra(data, acc, rbo->pos, scale, min_clamp, color, text_flag);
}
#endif

static void OVERLAY_colliding_face_on_box(OVERLAY_Data *data,
                                          Object *ob,
                                          const float color[4])
{
    float corr_rot[3][3];
    float final_mat[4][4];
    float loc[3] = {0.0f};
    float ax[3] = {0.0f};
    int face;
    RigidBodyOb *rbo = ob->rigidbody_object;
    for(int i=0; i<3; i++) {
       face = rbo->colliding_faces[i];
       zero_v3(loc);
       zero_v3(ax);
       if(face == -1) {
           continue;
       }
       switch(face) {
         case 0:
           ax[0] = 1.0f;
           axis_angle_to_mat3(corr_rot, ax, M_PI_2);
           loc[1] = -1.0f;
           break;
         case 1:
           ax[1] = 1.0f;
           axis_angle_to_mat3(corr_rot, ax, M_PI_2);
           loc[0] = 1.0f;
           break;
         case 2:
           unit_m3(corr_rot);
           loc[2] = -1.0f;
           break;
         case 3:
           ax[1] = 1.0f;
           axis_angle_to_mat3(corr_rot, ax, M_PI_2);
           loc[0] = -1.0f;
           break;
         case 4:
            unit_m3(corr_rot);
            loc[2] = 1.0f;
            break;
         case 5:
           ax[0] = 1.0f;
           axis_angle_to_mat3(corr_rot, ax, M_PI_2);
           loc[1] = 1.0f;
           break;
       }

       GPUBatch *geom = DRW_cache_quad_get();

       mul_m4_m4m3(final_mat, ob->obmat, corr_rot);
       mul_m4_v3(ob->obmat, loc);
       copy_v3_v3(final_mat[3], loc);
       final_mat[3][3] = 1.0f;

       GPUShader *sh = OVERLAY_shader_uniform_color();
       DRWShadingGroup *grp = DRW_shgroup_create(sh, data->psl->extra_ps[1]);
       DRW_shgroup_uniform_vec4_copy(grp, "color", color);
       DRW_shgroup_call_obmat(grp, geom, final_mat);
     }

}

static void OVERLAY_colliding_face_on_cylinder(OVERLAY_Data *data,
                                               Object *ob,
                                               const float color[4])
{
  int face;
  float loc[4];
  float final_mat[4][4] = {{0.0f}};
  for(int i=0; i<2; i++) {
    face = ob->rigidbody_object->colliding_faces[i];
    zero_v4(loc);
    if(face > -1) {
        switch(face){
          case 2:
            loc[2] = -1.0f;
            loc[3] = 1.0f;
            break;
          case 4:
            loc[2] = 1.0f;
            loc[3] = 1.0f;
            break;
        }

      GPUBatch *geom = DRW_cache_cylinder_face_get() ;
      printf("face%d\n",face);

      copy_m4_m4(final_mat, ob->obmat);
      mul_m4_v4(ob->obmat, loc);
      copy_v4_v4(final_mat[3], loc);

      GPUShader *sh = OVERLAY_shader_uniform_color();
      DRWShadingGroup *grp = DRW_shgroup_create(sh, data->psl->extra_ps[1]);
      DRW_shgroup_uniform_vec4_copy(grp, "color", color);
      DRW_shgroup_call_obmat(grp, geom, final_mat);
    }
  }
}

static void OVERLAY_colliding_face_on_cone(OVERLAY_Data *data,
                                           Object *ob,
                                           const float color[4])
{
  float loc[4] = {0.0f, 0.0f, -1.0f, 1.0f};
  float final_mat[4][4] = {{0.0f}};
  if(ob->rigidbody_object->colliding_faces[0] > -1) {
      GPUBatch *geom = DRW_cache_cone_face_get();

      copy_m4_m4(final_mat, ob->obmat);
      mul_m4_v4(ob->obmat, loc);
      copy_v4_v4(final_mat[3], loc);
      printf("yes");

      GPUShader *sh = OVERLAY_shader_uniform_color();
      DRWShadingGroup *grp = DRW_shgroup_create(sh, data->psl->extra_ps[1]);
      DRW_shgroup_uniform_vec4_copy(grp, "color", color);
      DRW_shgroup_call_obmat(grp, geom, final_mat);
  }
}

static void OVERLAY_indicate_collision(OVERLAY_Data *data, Object *ob)
{

  OVERLAY_PrivateData *pd = data->stl->pd;
  OVERLAY_ExtraCallBuffers *cb = &pd->extra_call_buffers[1];
  float mat[4][4];
  float dir[3];
  copy_v3_v3(dir, ob->rigidbody_object->norm_forces[0].vector);
  normalize_v3(dir);
  if (!is_zero_v3(ob->rigidbody_object->norm_forces[0].vector)) {
    float color[4] = {0.0f, 1.0f, 0.2f, 0.8f};
    switch (ob->rigidbody_object->shape) {
      case RB_SHAPE_BOX:
        OVERLAY_bounds(cb, ob, color, OB_BOUND_BOX, true, mat);
        OVERLAY_colliding_face_on_box(data, ob, color);
        break;
      case RB_SHAPE_SPHERE:
        OVERLAY_bounds(cb, ob, color, OB_BOUND_SPHERE, true, NULL);
        break;
      case RB_SHAPE_CONE:
        OVERLAY_bounds(cb, ob, color, OB_BOUND_CONE, true, mat);
        OVERLAY_colliding_face_on_cone(data, ob, color);
        break;
      case RB_SHAPE_CYLINDER:
        OVERLAY_bounds(cb, ob, color, OB_BOUND_CYLINDER, true, mat);
        OVERLAY_colliding_face_on_cylinder(data, ob, color);
        break;
      case RB_SHAPE_CAPSULE:
        OVERLAY_bounds(cb, ob, color, OB_BOUND_CAPSULE, true, NULL);
        break;
      case RB_SHAPE_CONVEXH:
        OVERLAY_non_primitive_collision_shape(cb, ob, color);
        break;
      case RB_SHAPE_TRIMESH:
        OVERLAY_non_primitive_collision_shape(cb, ob, color);
        break;
    }
  }
}

static void OVERLAY_linear_limits_single_object_rod(RigidBodyCon *rbc,
                                             Object *rb_ob,
                                             Object *constraint_ob,
                                             int axis,
                                             OVERLAY_ExtraCallBuffers *cb) {
    float translation_range;
    float color[4] = {0.0f, 1.0f, 0.2f, 1.0f};
    float line_start[3] = {0};
    float line_end[3] = {0};

    switch(axis){
       case 0:
        translation_range = (rbc->limit_lin_x_upper - rbc->limit_lin_x_lower);
        line_start[axis] += rbc->limit_lin_x_lower;
        line_end[axis] += rbc->limit_lin_x_upper;
        break;
       case 1:
        translation_range = (rbc->limit_lin_y_upper - rbc->limit_lin_y_lower);
        line_start[axis] += rbc->limit_lin_y_lower;
        line_end[axis] += rbc->limit_lin_y_upper;
        break;
       case 2:
        translation_range = (rbc->limit_lin_z_upper - rbc->limit_lin_z_lower);
        line_start[axis] += rbc->limit_lin_z_lower;
        line_end[axis] += rbc->limit_lin_z_upper;
        break;
    }
    if(fabsf(translation_range)>0.0f){
        float mat[3][3];
        mat4_to_rot(mat, constraint_ob->obmat);
        mul_m3_v3(mat,  line_start);
        mul_m3_v3(mat, line_end);
        add_v3_v3(line_start, rb_ob->loc);
        add_v3_v3(line_end, rb_ob->loc);
        OVERLAY_extra_line_dashed(cb, line_start, line_end, color);
    }
}

/* Draw constraint limits as walls. */
static void OVERLAY_linear_limits_walls(Object *ob,
                                         const float transform_mat[3][3],
                                         const float corr_rot[3][3],
                                         float upper_wall_pos[3],
                                         float lower_wall_pos[3],
                                         const float translation_range,
                                         OVERLAY_Data *data,
                                         bool fade_walls) {

    float color[4];
    copy_v4_v4(color, G_draw.block.colorLibrary);

    float distance_from_first_wall[3];
    float distance_from_second_wall[3];
    float *wall_pos;
    float ob_pos[3];
    float size[3];

    copy_v3_v3(ob_pos, (float *)ob->obmat[3]);
    sub_v3_v3v3(distance_from_first_wall, ob_pos, upper_wall_pos);
    sub_v3_v3v3(distance_from_second_wall, ob_pos, lower_wall_pos);

    float d1 = fabsf(len_v3(distance_from_first_wall));
    float d2 = fabsf(len_v3(distance_from_second_wall));

    float draw_range = translation_range < 12.0f ? (translation_range/2.0f)*0.5f : 3.0f;

    bool draw_upper_wall = (d1 <= draw_range) || !fade_walls;
    bool draw_lower_wall = (d2 <= draw_range) || !fade_walls;

    float final_mat[4][4] = {{0.0f}};
    copy_m4_m3(final_mat, transform_mat);
    mul_m4_m4m3(final_mat, final_mat, corr_rot);
    /* Scale the walls based on the size of ob1. */
    mat4_to_size(size, ob->obmat);
    mul_v3_fl(size, 1.5f);
    mul_v3_fl(final_mat[0], size[0]);
    mul_v3_fl(final_mat[1], size[1]);
    mul_v3_fl(final_mat[2], size[2]);

     if(fabsf(translation_range)>0.0f && draw_upper_wall) {
         wall_pos = upper_wall_pos ;
         float distance = d1;
         if(fade_walls) {
             mul_v4_fl(color, (draw_range - distance)/draw_range);
         }

         GPUBatch *geom = DRW_cache_quad_get();

         copy_v3_v3(final_mat[3], wall_pos);
         final_mat[3][3] = 1.0f;

         GPUShader *sh = OVERLAY_shader_uniform_color();
         DRWShadingGroup *grp = DRW_shgroup_create(sh, data->psl->extra_ps[0]);
         DRW_shgroup_uniform_vec4_copy(grp, "color", color);
         DRW_shgroup_call_obmat(grp, geom, final_mat);
     }
     if(fabsf(translation_range)>0.0f && draw_lower_wall) {
         wall_pos = lower_wall_pos ;
         float distance = d2;
         if(fade_walls) {
             mul_v4_fl(color, (draw_range - distance)/draw_range);
         }

         GPUBatch *geom = DRW_cache_quad_get();

         copy_v3_v3(final_mat[3], wall_pos);
         final_mat[3][3] = 1.0f;

         GPUShader *sh = OVERLAY_shader_uniform_color();
         DRWShadingGroup *grp = DRW_shgroup_create(sh, data->psl->extra_ps[0]);
         DRW_shgroup_uniform_vec4_copy(grp, "color", color);
         DRW_shgroup_call_obmat(grp, geom, final_mat);
     }
}

static void OVERLAY_linear_limits(RigidBodyCon *rbc,
                                             Object *ob1,
                                             Object *ob2,
                                             Object *constraint_ob,
                                             int axis,
                                             OVERLAY_ExtraCallBuffers *cb,
                                             OVERLAY_Data *data) {

    float color[4];
    copy_v4_v4(color, G_draw.block.colorLibrary);

    float line_start[3] = {0.0f};
    float line_end[3] = {0.0f};
    float ob1_projection[3] = {0.0f};
    float ob2_projection[3] = {0.0f};

    float ob1_upper_wall_pos[3] = {0.0f};
    float ob2_upper_wall_pos[3] = {0.0f};
    float ob1_lower_wall_pos[3] = {0.0f};
    float ob2_lower_wall_pos[3] = {0.0f};

    float constrained_axis[3] = {0.0f};
    constrained_axis[axis] = 1.0f;

    /* Correct initial orientation of walls. */
    float corr_rot[3][3];
    /* Constraint ob rotation. */
    float constraint_ob_rot[3][3] = {{0.0f}};
    /* Ob1 rotation. */
    float ob1_rot[3][3] = {{0.0f}};
    /* Ob1 frame obained from bullet. */
    float ob1_frame_offset[3][3] = {{0.0f}};
    /* Ob1 final transform matrix. */
    float transform_mat1[3][3] = {{0.0f}};
    /* Ob2 rotation. */
    float ob2_rot[3][3] = {{0.0f}};
    /* Ob1 frame obained from bullet. */
    float ob2_frame_offset[3][3] = {{0.0f}};
    /* Ob2 final transform matrix. */
    float transform_mat2[3][3] = {{0.0f}};
    /* Constraint origin in objects' frames. */
    float ob1_origin[3];
    float ob2_origin[3];
    /* Rigidbody positions. */
    float ob1_pos[3];
    float ob2_pos[3];

    copy_m3_m4(ob1_rot, ob1->obmat);
    copy_m3_m4(ob2_rot, ob2->obmat);
    copy_v3_v3(ob1_pos, (float *)ob1->obmat[3]);
    copy_v3_v3(ob2_pos, (float *)ob2->obmat[3]);
    Object *orig_constraint_ob = (Object *)constraint_ob->id.orig_id;

    BKE_object_rot_to_mat3(constraint_ob, constraint_ob_rot, false);
    if(orig_constraint_ob->rigidbody_constraint->physics_constraint != NULL ) {
      RB_constraint_get_transforms_slider(orig_constraint_ob->rigidbody_constraint->physics_constraint, ob1_frame_offset, ob2_frame_offset, ob1_origin, ob2_origin, NULL);
    }
    mul_m3_m3m3(transform_mat1, ob1_rot, ob1_frame_offset);
    mul_m3_m3m3(transform_mat2, ob2_rot, ob2_frame_offset);
    mul_m4_v3(ob1->obmat, ob1_origin);
    mul_m4_v3(ob2->obmat, ob2_origin);


    bool draw_rod = rbc->type == RBC_TYPE_SLIDER;
    bool is_constrained;

    float translation_range;
    /* Axis that lies along the plane perpendicular to constrained axis. */
    float ax[3] = {0.0f};

    switch(axis){
       case 0:
        translation_range = (rbc->limit_lin_x_upper - rbc->limit_lin_x_lower);
        ob2_upper_wall_pos[0] += (rbc->limit_lin_x_upper);
        ob2_lower_wall_pos[0] += (rbc->limit_lin_x_lower);
        ob1_upper_wall_pos[0] += (rbc->limit_lin_x_upper);
        ob1_lower_wall_pos[0] += (rbc->limit_lin_x_lower);
        ax[1] = 1.0f;
        axis_angle_to_mat3(corr_rot, ax, M_PI_2);
        is_constrained = (rbc->flag & RBC_FLAG_USE_LIMIT_LIN_X);
        break;
       case 1:
        translation_range = (rbc->limit_lin_y_upper - rbc->limit_lin_y_lower);
        ob2_upper_wall_pos[1] += (rbc->limit_lin_y_upper);
        ob2_lower_wall_pos[1] += (rbc->limit_lin_y_lower);
        ob1_upper_wall_pos[1] += (rbc->limit_lin_y_upper);
        ob1_lower_wall_pos[1] += (rbc->limit_lin_y_lower);
        ax[0] = 1.0f;
        axis_angle_to_mat3(corr_rot, ax, M_PI_2);
        is_constrained = (rbc->flag & RBC_FLAG_USE_LIMIT_LIN_Y);
        break;
       case 2:
        translation_range = (rbc->limit_lin_z_upper - rbc->limit_lin_z_lower);
        ob2_upper_wall_pos[2] += (rbc->limit_lin_z_upper);
        ob2_lower_wall_pos[2] += (rbc->limit_lin_z_lower);
        ob1_upper_wall_pos[2] += (rbc->limit_lin_z_upper);
        ob1_lower_wall_pos[2] += (rbc->limit_lin_z_lower);
        /* Artbitrary axis so that the matrix(corr_rot) doesn't remain empty. */
        ax[0] = 1.0f;
        axis_angle_to_mat3(corr_rot, ax, 0.0f);
        is_constrained = (rbc->flag & RBC_FLAG_USE_LIMIT_LIN_Z);
        break;
    }

    if(!is_constrained && !draw_rod) {
        return;
    }
    mul_m3_v3(transform_mat1, constrained_axis);
    float mid_point[3];
    float rod_partA[3];
    float rod_partB[3];
    float mid_point_to_wall1[3];
    float mid_point_to_wall2[3];

    add_v3_v3v3(mid_point, ob1_pos, ob2_pos);
    mul_v3_fl(mid_point, 0.5f);

    if(is_constrained) {
      float delta[3];
      sub_v3_v3v3(delta, ob2_origin, ob1_origin);
      for(int i=0; i<3; i++) {
        if(i == axis) {
            delta[i] = dot_v3v3(delta, constrained_axis);
        }
        else {
            delta[i] = 0.0f;
        }
      }
      sub_v3_v3(ob2_upper_wall_pos, delta);
      sub_v3_v3(ob2_lower_wall_pos, delta);
      add_v3_v3(ob1_upper_wall_pos, delta);
      add_v3_v3(ob1_lower_wall_pos, delta);

      mul_m3_v3(transform_mat1, ob2_upper_wall_pos);
      mul_m3_v3(transform_mat1, ob1_lower_wall_pos);
      mul_m3_v3(transform_mat1, ob1_upper_wall_pos);
      mul_m3_v3(transform_mat1, ob2_lower_wall_pos);

      add_v3_v3(ob2_upper_wall_pos, ob2_pos);
      add_v3_v3(ob2_lower_wall_pos, ob2_pos);
      add_v3_v3(ob1_upper_wall_pos, ob1_pos);
      add_v3_v3(ob1_lower_wall_pos, ob1_pos);
      add_v3_v3(line_start, mid_point);
      add_v3_v3(line_end, mid_point);


      if(len_v3v3(ob1_lower_wall_pos, ob2_upper_wall_pos)> len_v3v3(ob1_upper_wall_pos, ob2_lower_wall_pos)) {
          sub_v3_v3v3(mid_point_to_wall1, mid_point, ob2_upper_wall_pos);
          sub_v3_v3v3(mid_point_to_wall2, mid_point, ob1_lower_wall_pos);
      }
      else{
          sub_v3_v3v3(mid_point_to_wall1, mid_point, ob2_lower_wall_pos);
          sub_v3_v3v3(mid_point_to_wall2, mid_point, ob1_upper_wall_pos);
      }
    }
    else {
      sub_v3_v3v3(mid_point_to_wall1, mid_point, ob2_pos);
      sub_v3_v3v3(mid_point_to_wall2, mid_point, ob1_pos);
      add_v3_v3(line_start, mid_point);
      add_v3_v3(line_end, mid_point);
    }
    mul_v3_v3fl(rod_partB, constrained_axis, dot_v3v3(constrained_axis, mid_point_to_wall1));
    mul_v3_v3fl(rod_partA, constrained_axis, dot_v3v3(constrained_axis, mid_point_to_wall2));

    add_v3_v3(line_start, rod_partA);
    add_v3_v3(line_end, rod_partB);

    closest_to_line_segment_v3(ob1_projection, ob1_pos, line_start, line_end);
    closest_to_line_segment_v3(ob2_projection, ob2_pos, line_start, line_end);

    if(draw_rod) {
      OVERLAY_extra_line_dashed(cb, line_start, line_end, color);
      OVERLAY_extra_line_dashed(cb, ob1_pos, ob1_projection, color);
      OVERLAY_extra_line_dashed(cb, ob2_pos, ob2_projection, color);
    }

    if(is_constrained) {

      bool fade_walls = rbc->flag & RBC_FLAG_DEBUG_DRAW_FADE_WALLS;

      if((ob1->rigidbody_object->type == RBO_TYPE_ACTIVE) && !(ob1->rigidbody_object->flag & RBO_FLAG_KINEMATIC)) {
        OVERLAY_linear_limits_walls(ob1, transform_mat1, corr_rot, ob1_upper_wall_pos, ob1_lower_wall_pos, translation_range, data, fade_walls);
      }
      if((ob2->rigidbody_object->type == RBO_TYPE_ACTIVE) && !(ob2->rigidbody_object->flag & RBO_FLAG_KINEMATIC)) {
        OVERLAY_linear_limits_walls(ob2, transform_mat2, corr_rot, ob2_upper_wall_pos, ob2_lower_wall_pos, translation_range, data, fade_walls);
      }
    }
}

static void OVERLAY_angular_limits_rods( OVERLAY_ExtraCallBuffers *cb,
                                     Object *ob,
                                     const float pivot[3],
                                     const float ob_offset_vec[3],
                                     const float coupled_ob_offset_vec[3],
                                     const float color[4]) {
    float start[3];
    float end[3];
    float mid[3];

    float ob_pivot_dist_vec[3];
    float ob_pos[3];
    copy_v3_v3(ob_pos, (float *)ob->obmat[3]);
    sub_v3_v3v3(ob_pivot_dist_vec, ob_pos, pivot);

    copy_v3_v3(start, ob_pos);
    copy_v3_v3(mid, pivot);
    copy_v3_v3(end, pivot);

    add_v3_v3(mid, ob_offset_vec);
    add_v3_v3(end, coupled_ob_offset_vec);

    float line_seg[3];
    sub_v3_v3v3(line_seg, mid, start);
    if(len_v3(line_seg)<1.0f && len_v3(line_seg)>0.000001f) {
        normalize_v3(line_seg);
        copy_v3_v3(start, mid);
        sub_v3_v3(start, line_seg);
    }

    OVERLAY_extra_line_dashed(cb, start, mid, color);
    OVERLAY_extra_line_dashed(cb, mid, end, color);
}

static void OVERLAY_angular_limits_disk(OVERLAY_Data *data,
                                     RigidBodyCon *rbc,
                                     Object *ob,
                                     const float transform_mat[3][3],
                                     const float corr_rot[3][3],
                                     const float angle,
                                     float angular_offset,
                                     const float delta_angle,
                                     const float axis_in_w[3],
                                     const float offset_vec[3],
                                     const float real_pivot[3],
                                     const float color[4]) {

    float ob_pivot_dist[3];
    float ob_pos[3];
    copy_v3_v3(ob_pos, (float *)ob->obmat[3]);
    sub_v3_v3v3(ob_pivot_dist, real_pivot, ob_pos);

    float final_transform_mat[3][3];
    mul_m3_m3m3(final_transform_mat, transform_mat, corr_rot);

    /* Rotate the arc about constrained axis by required amount.
     * The arc should lie along the object-pivot distance vector. */
    float constrained_axis_rot;

    if(ob == rbc->ob1) {
        float arc_start[3] = {cosf(0.0f+delta_angle), sinf(0.0f+delta_angle), 0.0f};
        mul_m3_v3(final_transform_mat, arc_start);
        constrained_axis_rot = angle_signed_on_axis_v3v3_v3(arc_start, ob_pivot_dist, axis_in_w);
    }
    else {
        float arc_start1[3] = {cosf(angle + (2*angular_offset)-delta_angle), sinf(angle + (2*angular_offset)-delta_angle), 0.0f};
        mul_m3_v3(final_transform_mat, arc_start1);
        constrained_axis_rot = angle_signed_on_axis_v3v3_v3(arc_start1, ob_pivot_dist, axis_in_w);
    }
    angular_offset += (M_PI - constrained_axis_rot);

    float disk_pos[3];
    copy_v3_v3(disk_pos, real_pivot);
    add_v3_v3(disk_pos, offset_vec);

    float final_mat[4][4] = {{0.0f}};

    copy_m4_m3(final_mat, final_transform_mat);

    copy_v3_v3(final_mat[3], disk_pos);
    final_mat[3][3] = 1.0f;

    float segment_width = M_PI*10/180;
    int n_segments = floor(angle/segment_width) + 2;

    GPUShader *sh = OVERLAY_shader_constraint_angular_limits();
    DRWShadingGroup *grp = DRW_shgroup_create(sh, data->psl->extra_ps[1]);
    DRW_shgroup_uniform_vec4_array_copy(grp, "mat_vecs", final_mat, 4);
    DRW_shgroup_uniform_float_copy(grp, "offset", angular_offset);
    DRW_shgroup_uniform_float_copy(grp, "angle", angle);
    DRW_shgroup_uniform_vec3_copy(grp, "color", color);
    DRW_shgroup_call_procedural_lines(grp, NULL, n_segments);
}

static void OVERLAY_angular_limits(OVERLAY_Data *data,
                                   OVERLAY_ExtraCallBuffers *cb,
                                     RigidBodyCon *rbc,
                                     Object *ob1,
                                     Object *ob2,
                                     Object *constraint_ob,
                                     int axis) {

    float color1[4] = {0.0f, 1.0f, 0.2f, 1.0f};
    float color2[4] = {0.0f, 0.5f, 0.8f, 1.0f};
    float angle = 0.0f;
    float angular_offset = 0.0f;

    float t1[3][3] = {{0.0f}};
    float t2[3][3] = {{0.0f}};
    float ob1_rot[3][3] = {{0.0f}};
    float ob2_rot[3][3] = {{0.0f}};
    float ob1_basis[3][3] = {{0.0f}};
    float ob1_origin[3] = {0.0f};
    float ob2_basis[3][3] = {{0.0f}};
    float ob2_origin[3] = {0.0f};

    float ob1_pos[3];
    float ob2_pos[3];

    bool draw_disks;

    copy_m3_m4(ob1_rot, ob1->obmat);
    copy_m3_m4(ob2_rot, ob2->obmat);
    copy_v3_v3(ob1_pos, (float *)ob1->obmat[3]);
    copy_v3_v3(ob2_pos, (float *)ob2->obmat[3]);

    Object *orig_constraint_ob = (Object *)constraint_ob->id.orig_id;

    if(rbc->type == RBC_TYPE_HINGE && orig_constraint_ob->rigidbody_constraint->physics_constraint != NULL ) {
        RB_constraint_get_transforms_hinge(orig_constraint_ob->rigidbody_constraint->physics_constraint, ob1_basis, ob2_basis, ob1_origin, ob2_origin);

    }
    else if(rbc->type == RBC_TYPE_PISTON && orig_constraint_ob->rigidbody_constraint->physics_constraint != NULL ) {
        RB_constraint_get_transforms_slider(orig_constraint_ob->rigidbody_constraint->physics_constraint, ob1_basis, ob2_basis, ob1_origin, ob2_origin, NULL);

    }
    mul_m3_m3m3(t1, ob1_rot, ob1_basis);
    mul_m3_m3m3(t2, ob2_rot, ob2_basis);

    float real_pivot[3] = {0.0f};
    float ob1_initial_rot_inv[3][3] = {{0.0f}};
    BKE_object_rot_to_mat3(ob1, ob1_initial_rot_inv, false);
    invert_m3(ob1_initial_rot_inv);
    if(constraint_ob == ob1) {
      copy_v3_v3(real_pivot, ob1_pos);
    }
    else if(constraint_ob == ob2) {
      copy_v3_v3(real_pivot, ob2_pos);
    }
    else{
      mul_v3_m4v3(real_pivot, ob1->obmat, ob1_origin);
    }
    /* Rotate disk to correct initial orientation */
    float corr_rot[3][3] = {{0}};
    float constrained_axis_w[3] = {0.0f};

    constrained_axis_w[axis] = 1.0f;

    mul_m3_v3(t1, constrained_axis_w);
    mul_m3_v3(ob1_rot, constrained_axis_w);

    float ax[3] = {0.0f};

    switch(axis){
       case 0:
        angle = (rbc->limit_ang_x_upper - rbc->limit_ang_x_lower);
        angular_offset = rbc->limit_ang_x_lower;
        ax[1] = 1.0f;
        axis_angle_to_mat3(corr_rot, ax, M_PI_2);
        draw_disks = (rbc->flag & RBC_FLAG_USE_LIMIT_ANG_X);
        break;
       case 1:
        angle = (rbc->limit_ang_y_upper - rbc->limit_ang_y_lower);
        angular_offset = rbc->limit_ang_y_lower;
        ax[0] = 1.0f;
        axis_angle_to_mat3(corr_rot, ax, M_PI_2);
        draw_disks = (rbc->flag & RBC_FLAG_USE_LIMIT_ANG_Y);
        break;
       case 2:
        angle = (rbc->limit_ang_z_upper - rbc->limit_ang_z_lower);
        angular_offset = rbc->limit_ang_z_lower;
        ax[0] = 1.0f;
        unit_m3(corr_rot);
        draw_disks = (rbc->flag & RBC_FLAG_USE_LIMIT_ANG_Z);
        break;
    }


    float ob1_pivot_dist[3];
    float ob2_pivot_dist[3];
    if(ob1->rigidbody_object->type == RBO_TYPE_ACTIVE) {
      sub_v3_v3v3(ob1_pivot_dist, ob1_pos, real_pivot);
    }
    else {
      zero_v3(ob1_pivot_dist);
    }
    if(ob2->rigidbody_object->type == RBO_TYPE_ACTIVE) {
      sub_v3_v3v3(ob2_pivot_dist, ob2_pos, real_pivot);
    }
    else {
      zero_v3(ob2_pivot_dist);
    }
    float ob1_offset = dot_v3v3(ob1_pivot_dist, constrained_axis_w);
    float ob1_offset_vec[3];
    float ob2_offset = dot_v3v3(ob2_pivot_dist, constrained_axis_w);
    float ob2_offset_vec[3];
    copy_v3_v3(ob1_offset_vec, constrained_axis_w);
    copy_v3_v3(ob2_offset_vec, constrained_axis_w);
    mul_v3_fl(ob1_offset_vec, ob1_offset);
    mul_v3_fl(ob2_offset_vec, ob2_offset);

    float axis_ob1[3];
    float axis_ob2[3];
    copy_v3_v3(axis_ob1, ax);
    copy_v3_v3(axis_ob2, ax);
    mul_m3_v3(t1, axis_ob1);
    mul_m3_v3(t2, axis_ob2);
    float delta_angle = angle_signed_on_axis_v3v3_v3(axis_ob1, axis_ob2, constrained_axis_w);

    OVERLAY_PrivateData *pd = data->stl->pd;
    cb = &pd->extra_call_buffers[1];

    if(ob1->rigidbody_object->type == RBO_TYPE_ACTIVE) {
      if(draw_disks) {
        OVERLAY_angular_limits_disk(data, rbc, ob1, t2, corr_rot, angle, angular_offset, delta_angle, constrained_axis_w, ob1_offset_vec, real_pivot, color1);
      }
      OVERLAY_angular_limits_rods(cb, ob1, real_pivot, ob1_offset_vec, ob2_offset_vec, color2);
    }
    if(ob2->rigidbody_object->type == RBO_TYPE_ACTIVE) {
      if(draw_disks) {
        OVERLAY_angular_limits_disk(data, rbc, ob2, t1, corr_rot, angle, angular_offset, delta_angle, constrained_axis_w, ob2_offset_vec, real_pivot, color2);
      }
      OVERLAY_angular_limits_rods(cb, ob2, real_pivot, ob2_offset_vec, ob1_offset_vec, color1);
    }
}

static void OVERLAY_rb_constraint_fixed(OVERLAY_ExtraCallBuffers *cb, Object *ob1, Object*ob2) {

    float color[4] = {0.0f, 0.5f, 0.8f, 1.0f};
    float ob1_pos[3];
    float ob2_pos[3];
    for(int i=0; i<3; i++) {
        ob1_pos[i] = ob1->obmat[3][i];
        ob2_pos[i] = ob2->obmat[3][i];
    }

    OVERLAY_extra_line_dashed(cb, ob1_pos, ob2_pos, color);
}

static void OVERLAY_rb_constraint_limits(OVERLAY_ExtraCallBuffers *cb, OVERLAY_Data *data, Object *ob){
    RigidBodyCon *rbc = ob->rigidbody_constraint;
    Object *ob1 = rbc->ob1;
    Object *ob2 = rbc->ob2;

    if(ob1==NULL || ob2==NULL) {
        return;
    }

    if(ob1->rigidbody_object->type == RBO_TYPE_PASSIVE && ob2->rigidbody_object->type == RBO_TYPE_PASSIVE) {
        return;
    }

    if((ob1->rigidbody_object->flag & RBO_FLAG_KINEMATIC) && (ob2->flag & RBO_FLAG_KINEMATIC)) {
        return;
    }

    bool is_rod_type_constraint = ((ob1->rigidbody_object->type == RBO_TYPE_ACTIVE &&
                                  (ob2->rigidbody_object->type == RBO_TYPE_PASSIVE)) ||
                                  (ob2->rigidbody_object->type == RBO_TYPE_ACTIVE &&
                                  (ob1->rigidbody_object->type == RBO_TYPE_PASSIVE))) &&
                                  (rbc->type == RBC_TYPE_SLIDER);

    if(is_rod_type_constraint) {
      Object *rb_ob = ob1->rigidbody_object->type == RBO_TYPE_ACTIVE ? ob1 : ob2;
          OVERLAY_linear_limits_single_object_rod(rbc, rb_ob, ob, 0, cb);
    }
    else {
      switch(rbc->type) {
        case RBC_TYPE_FIXED:
          OVERLAY_rb_constraint_fixed(cb, ob1, ob2);
          break;
        case RBC_TYPE_HINGE:
          OVERLAY_angular_limits(data, cb, rbc, ob1, ob2, ob, 2);
          break;
        case RBC_TYPE_SLIDER:
          OVERLAY_linear_limits(rbc, ob1, ob2, ob, 0, cb, data);
          break;
        case RBC_TYPE_6DOF:
          break;
        case RBC_TYPE_6DOF_SPRING:
          break;
        case RBC_TYPE_PISTON:
          OVERLAY_linear_limits(rbc, ob1, ob2, ob, 0, cb, data);
          OVERLAY_angular_limits(data, cb, rbc, ob1, ob2, ob, 0);
       }
    }
}

void OVERLAY_extra_cache_populate(OVERLAY_Data *vedata, Object *ob)
{
  OVERLAY_ExtraCallBuffers *cb = OVERLAY_extra_call_buffer_get(vedata, ob);
  OVERLAY_PrivateData *pd = vedata->stl->pd;
  const DRWContextState *draw_ctx = DRW_context_state_get();
  ViewLayer *view_layer = draw_ctx->view_layer;
  Scene *scene = draw_ctx->scene;
  ModifierData *md = NULL;

  const bool is_select_mode = DRW_state_is_select();
  const bool is_paint_mode = (draw_ctx->object_mode &
                              (OB_MODE_ALL_PAINT | OB_MODE_ALL_PAINT_GPENCIL)) != 0;
  const bool from_dupli = (ob->base_flag & (BASE_FROM_SET | BASE_FROM_DUPLI)) != 0;
  const bool has_bounds = !ELEM(ob->type, OB_LAMP, OB_CAMERA, OB_EMPTY, OB_SPEAKER, OB_LIGHTPROBE);
  const bool has_texspace = has_bounds &&
                            !ELEM(ob->type, OB_EMPTY, OB_LATTICE, OB_ARMATURE, OB_GPENCIL);

  const bool draw_relations = ((pd->v3d_flag & V3D_HIDE_HELPLINES) == 0) && !is_select_mode;
  const bool draw_obcenters = !is_paint_mode &&
                              (pd->overlay.flag & V3D_OVERLAY_HIDE_OBJECT_ORIGINS) == 0;
  const bool draw_texspace = (ob->dtx & OB_TEXSPACE) && has_texspace;
  const bool draw_obname = (ob->dtx & OB_DRAWNAME) && DRW_state_show_text();
  const bool draw_bounds = has_bounds && ((ob->dt == OB_BOUNDBOX) ||
                                          ((ob->dtx & OB_DRAWBOUNDOX) && !from_dupli));
  const bool draw_xform = draw_ctx->object_mode == OB_MODE_OBJECT &&
                          (scene->toolsettings->transform_flag & SCE_XFORM_DATA_ORIGIN) &&
                          (ob->base_flag & BASE_SELECTED) && !is_select_mode;
  /* Don't show fluid domain overlay extras outside of cache range. */
  const bool draw_volume = !from_dupli &&
                           (md = BKE_modifiers_findby_type(ob, eModifierType_Fluid)) &&
                           (BKE_modifier_is_enabled(scene, md, eModifierMode_Realtime)) &&
                           (((FluidModifierData *)md)->domain != NULL) &&
                           (CFRA >= (((FluidModifierData *)md)->domain->cache_frame_start)) &&
                           (CFRA <= (((FluidModifierData *)md)->domain->cache_frame_end));

  float *color;
  int theme_id = DRW_object_wire_theme_get(ob, view_layer, &color);

  if (ob->pd && ob->pd->forcefield) {
    OVERLAY_forcefield(cb, ob, view_layer);
  }

  if (draw_bounds) {
    OVERLAY_bounds(cb, ob, color, ob->boundtype, false, NULL);
  }
  /* Helpers for when we're transforming origins. */
  if (draw_xform) {
    const float color_xform[4] = {0.15f, 0.15f, 0.15f, 0.7f};
    DRW_buffer_add_entry(cb->origin_xform, color_xform, ob->obmat);
  }
  /* don't show object extras in set's */
  if (!from_dupli) {
    if (draw_obcenters) {
      OVERLAY_object_center(cb, ob, pd, view_layer);
    }
    if (draw_relations) {
      OVERLAY_relationship_lines(cb, draw_ctx->depsgraph, draw_ctx->scene, ob);
    }
    if (draw_obname) {
      OVERLAY_object_name(ob, theme_id);
    }
    if (draw_texspace) {
      OVERLAY_texture_space(cb, ob, color);
    }
    if (ob->rigidbody_object != NULL) {
      if (!is_zero_v3(ob->rigidbody_object->norm_forces[0].vector) &&
          ob->rigidbody_object->sim_display_options & RB_SIM_COLLISIONS) {
        OVERLAY_indicate_collision(vedata, ob);
      }
      else {
        OVERLAY_collision(cb, ob, color);
      }
#ifdef WITH_BULLET
      if (ob->rigidbody_object->sim_display_options & RB_SIM_FORCES)
        OVERLAY_forces_extra(vedata, scene, ob->rigidbody_object);
      if (ob->rigidbody_object->sim_display_options & RB_SIM_ACCELERATION)
        OVERLAY_acceleration_extra(
            vedata, ob->rigidbody_object, draw_ctx->depsgraph, draw_ctx->scene);
      if (ob->rigidbody_object->sim_display_options & RB_SIM_VELOCITY)
        OVERLAY_velocity_extra(vedata, ob->rigidbody_object);
#endif
    }

    if(ob->rigidbody_constraint != NULL) {
        if(ob->rigidbody_constraint->flag & RBC_FLAG_DEBUG_DRAW_LIMITS) {
          OVERLAY_rb_constraint_limits(cb, vedata, ob);
        }
    }

    if (ob->dtx & OB_AXIS) {
      DRW_buffer_add_entry(cb->empty_axes, color, ob->obmat);
    }
    if (draw_volume) {
      OVERLAY_volume_extra(cb, vedata, ob, md, scene, color);
    }
  }
}

void OVERLAY_extra_blend_draw(OVERLAY_Data *vedata)
{
  DRW_draw_pass(vedata->psl->extra_blend_ps);
}

void OVERLAY_extra_draw(OVERLAY_Data *vedata)
{
  DRW_draw_pass(vedata->psl->extra_ps[0]);
}

void OVERLAY_extra_in_front_draw(OVERLAY_Data *vedata)
{
  DRW_draw_pass(vedata->psl->extra_ps[1]);

  OVERLAY_volume_free_smoke_textures(vedata);
}

void OVERLAY_extra_centers_draw(OVERLAY_Data *vedata)
{
  OVERLAY_PassList *psl = vedata->psl;

  DRW_draw_pass(psl->extra_grid_ps);
  DRW_draw_pass(psl->extra_centers_ps);
}
