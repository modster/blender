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
 * An instance contains all structures needed to do a complete render.
 */

#include "BKE_global.h"
#include "BKE_object.h"
#include "BLI_rect.h"
#include "DEG_depsgraph_query.h"
#include "DNA_ID.h"
#include "DNA_lightprobe_types.h"
#include "DNA_modifier_types.h"

#include "eevee_instance.hh"

namespace blender::eevee {

/* -------------------------------------------------------------------- */
/** \name Init
 *
 * Init funcions need to be called once at the start of a frame.
 * Active camera, render extent and enabled render passes are immutable until next init.
 * This takes care of resizing output buffers and view in case a parameter changed.
 * IMPORTANT: xxx.init() functions are NOT meant to acquire and allocate DRW resources.
 * Any attempt to do so will likely produce use after free situations.
 * \{ */

void Instance::init(const ivec2 &output_res,
                    const rcti *output_rect,
                    RenderEngine *render_,
                    Depsgraph *depsgraph_,
                    const struct LightProbe *light_probe_,
                    Object *camera_object_,
                    const RenderLayer *render_layer_,
                    const DRWView *drw_view_,
                    const View3D *v3d_,
                    const RegionView3D *rv3d_)
{
  render = render_;
  depsgraph = depsgraph_;
  render_layer = render_layer_;
  camera_orig_object = camera_object_;
  drw_view = drw_view_;
  v3d = v3d_;
  rv3d = rv3d_;
  baking_probe = light_probe_;

  debug_mode = (eDebugMode)G.debug_value;

  update_eval_members();

  rcti render_border = output_crop(output_res, output_rect);

  /* Needs to be first. */
  sampling.init(scene);

  camera.init();
  motion_blur.init();
  render_passes.init(output_res, &render_border);
  main_view.init(output_res);
  velocity.init();
  shadows.init();
  lightprobes.init();
  lookdev.init(output_res, &render_border);
}

rcti Instance::output_crop(const int res[2], const rcti *crop)
{
  rcti rect;
  BLI_rcti_init(&rect, 0, res[0], 0, res[1]);
  /* Clip the render border to region bounds. */
  BLI_rcti_isect(crop, &rect, &rect);
  if (BLI_rcti_is_empty(&rect)) {
    BLI_rcti_init(&rect, 0, res[0], 0, res[1]);
  }
  return rect;
}

void Instance::set_time(float time)
{
  BLI_assert(render);
  DRW_render_set_time(render, depsgraph, floorf(time), fractf(time));
  update_eval_members();
}

void Instance::update_eval_members(void)
{
  scene = DEG_get_evaluated_scene(depsgraph);
  view_layer = DEG_get_evaluated_view_layer(depsgraph);
  camera_eval_object = (camera_orig_object) ?
                           DEG_get_evaluated_object(depsgraph, camera_orig_object) :
                           nullptr;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Sync
 *
 * Sync will gather data from the scene that can change over a time step (i.e: motion steps).
 * IMPORTANT: xxx.sync() functions area responsible for creating DRW resources (i.e: DRWView) as
 * well as querying temp texture pool. All DRWPasses should be ready by the end end_sync().
 * \{ */

void Instance::begin_sync()
{
  camera.sync();
  render_passes.sync();
  shading_passes.sync();
  main_view.sync();
  world.sync();
  raytracing.sync();
  hiz.sync();

  lookdev.sync_background();
  lookdev.sync_overlay();

  materials.begin_sync();
  velocity.begin_sync();
  lights.begin_sync();
  shadows.begin_sync();
  lightprobes.begin_sync();
}

void Instance::object_sync(Object *ob)
{
  const bool is_renderable_type = ELEM(
      ob->type, OB_MESH, OB_CURVE, OB_SURF, OB_FONT, OB_MBALL, OB_LAMP, OB_VOLUME, OB_GPENCIL);
  const int ob_visibility = DRW_object_visibility_in_active_context(ob);
  const bool partsys_is_visible = (ob_visibility & OB_VISIBLE_PARTICLES) != 0 &&
                                  (ob->type == OB_MESH);
  const bool object_is_visible = DRW_object_is_renderable(ob) &&
                                 (ob_visibility & OB_VISIBLE_SELF) != 0;

  if (!is_renderable_type || (!partsys_is_visible && !object_is_visible)) {
    return;
  }

  ObjectHandle &ob_handle = sync.sync_object(ob);

  if (partsys_is_visible && ob != DRW_context_state_get()->object_edit) {
    LISTBASE_FOREACH (ModifierData *, md, &ob->modifiers) {
      if (md->type == eModifierType_ParticleSystem) {
        hair_sync(ob, ob_handle, md);
      }
    }
  }

  if (object_is_visible) {
    switch (ob->type) {
      case OB_LAMP:
        lights.sync_light(ob, ob_handle);
        break;
      case OB_MESH:
      case OB_CURVE:
      case OB_SURF:
      case OB_FONT:
      case OB_MBALL: {
        mesh_sync(ob, ob_handle);
        break;
      }
      case OB_VOLUME:
        shading_passes.deferred.volume_add(ob);
        break;
      case OB_HAIR:
        hair_sync(ob, ob_handle);
        break;
      case OB_GPENCIL:
        gpencil_sync(ob, ob_handle);
        break;
      default:
        break;
    }
  }

  ob_handle.reset_recalc_flag();
}

/* Wrapper to use with DRW_render_object_iter. */
void Instance::object_sync_render(void *instance_,
                                  Object *ob,
                                  RenderEngine *engine,
                                  Depsgraph *depsgraph)
{
  UNUSED_VARS(engine, depsgraph);

  Instance &inst = *reinterpret_cast<Instance *>(instance_);

  if (inst.baking_probe != nullptr) {
    if (inst.baking_probe->visibility_grp != nullptr) {
      bool test = BKE_collection_has_object_recursive(inst.baking_probe->visibility_grp, ob);
      test = (inst.baking_probe->flag & LIGHTPROBE_FLAG_INVERT_GROUP) ? !test : test;
      if (!test) {
        return;
      }
    }
    /* Exclude planar lightprobes. */
    if (ob->type == OB_LIGHTPROBE) {
      LightProbe *prb = (LightProbe *)ob->data;
      if (prb->type == LIGHTPROBE_TYPE_PLANAR) {
        return;
      }
    }
  }
  inst.object_sync(ob);
}

void Instance::end_sync(void)
{
  velocity.end_sync();
  lights.end_sync();
  sampling.end_sync();
  render_passes.end_sync();
  lightprobes.end_sync();
  subsurface.end_sync();
}

void Instance::render_sync(void)
{
  DRW_cache_restart();

  this->begin_sync();
  DRW_render_object_iter(this, render, depsgraph, object_sync_render);
  this->end_sync();

  DRW_render_instance_buffer_finish();
  /* Also we weed to have a correct fbo bound for DRW_hair_update */
  // GPU_framebuffer_bind();
  // DRW_hair_update();
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Rendering
 * \{ */

/**
 * Conceptually renders one sample per pixel.
 * Everything based on random sampling should be done here (i.e: DRWViews jitter)
 **/
void Instance::render_sample(void)
{
  if (sampling.finished()) {
    return;
  }

  /* Motion blur may need to do re-sync after a certain number of sample. */
  if (sampling.do_render_sync()) {
    this->render_sync();
  }

  sampling.step();

  /* TODO update shadowmaps, planars, etc... */
  // shadow_view_.render();

  main_view.render();

  motion_blur.step();
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Interface
 * \{ */

void Instance::render_frame(RenderLayer *render_layer, const char *view_name)
{
  while (!sampling.finished()) {
    this->render_sample();
    /* TODO(fclem) print progression. */
  }

  render_passes.read_result(render_layer, view_name);
}

void Instance::draw_viewport(DefaultFramebufferList *dfbl)
{
  this->render_sample();

  render_passes.resolve_viewport(dfbl);

  if (!sampling.finished_viewport()) {
    DRW_viewport_request_redraw();
  }
}

bool Instance::finished(void) const
{
  return sampling.finished();
}

/** \} */

}  // namespace blender::eevee
