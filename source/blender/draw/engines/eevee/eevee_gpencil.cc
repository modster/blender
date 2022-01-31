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
 */

#include "BKE_gpencil.h"
#include "BKE_object.h"
#include "DEG_depsgraph_query.h"
#include "DNA_gpencil_types.h"

#include "eevee_instance.hh"

namespace blender::eevee {

#define DO_BATCHING true

struct gpIterData {
  Instance &inst;
  Object *ob;
  MaterialArray &material_array;
  int cfra;

  /* Drawcall batching. */
  GPUBatch *geom = nullptr;
  Material *material = nullptr;
  int vfirst = 0;
  int vcount = 0;
  bool instancing = false;

  gpIterData(Instance &inst_, Object *ob_)
      : inst(inst_), ob(ob_), material_array(inst_.materials.material_array_get(ob_))
  {
    cfra = DEG_get_ctime(inst.depsgraph);
  };
};

static void gpencil_drawcall_flush(gpIterData &iter)
{
  if (iter.geom != NULL) {
    shgroup_geometry_call(iter.material->shading.shgrp,
                          iter.ob,
                          iter.geom,
                          iter.vfirst,
                          iter.vcount,
                          iter.instancing);
    shgroup_geometry_call(iter.material->prepass.shgrp,
                          iter.ob,
                          iter.geom,
                          iter.vfirst,
                          iter.vcount,
                          iter.instancing);
    shgroup_geometry_call(iter.material->shadow.shgrp,
                          iter.ob,
                          iter.geom,
                          iter.vfirst,
                          iter.vcount,
                          iter.instancing);
  }
  iter.geom = NULL;
  iter.vfirst = -1;
  iter.vcount = 0;
}

/* Group draw-calls that are consecutive and with the same type. Reduces GPU driver overhead. */
static void gpencil_drawcall_add(gpIterData &iter,
                                 GPUBatch *geom,
                                 Material *material,
                                 int v_first,
                                 int v_count,
                                 bool instancing)
{
  int last = iter.vfirst + iter.vcount;
  /* Interrupt draw-call grouping if the sequence is not consecutive. */
  if (!DO_BATCHING || (geom != iter.geom) || (material != iter.material) || (v_first - last > 3)) {
    gpencil_drawcall_flush(iter);
  }
  iter.geom = geom;
  iter.material = material;
  iter.instancing = instancing;
  if (iter.vfirst == -1) {
    iter.vfirst = v_first;
  }
  iter.vcount = v_first + v_count - iter.vfirst;
}

static void gpencil_stroke_sync(bGPDlayer *UNUSED(gpl),
                                bGPDframe *UNUSED(gpf),
                                bGPDstroke *gps,
                                void *thunk)
{
  gpIterData &iter = *(gpIterData *)thunk;

  Material *material = iter.material_array.materials[gps->mat_nr];
  MaterialGPencilStyle *gp_style = BKE_gpencil_material_settings(iter.ob, gps->mat_nr + 1);

  bool hide_material = (gp_style->flag & GP_MATERIAL_HIDE) != 0;
  bool show_stroke = ((gp_style->flag & GP_MATERIAL_STROKE_SHOW) != 0) ||
                     (!DRW_state_is_image_render() && ((gps->flag & GP_STROKE_NOFILL) != 0));
  bool show_fill = (gps->tot_triangles > 0) && ((gp_style->flag & GP_MATERIAL_FILL_SHOW) != 0);

  if (hide_material) {
    return;
  }

  if (show_fill) {
    GPUBatch *geom = DRW_cache_gpencil_fills_get(iter.ob, iter.cfra);
    int vfirst = gps->runtime.fill_start * 3;
    int vcount = gps->tot_triangles * 3;
    gpencil_drawcall_add(iter, geom, material, vfirst, vcount, false);
  }

  if (show_stroke) {
    GPUBatch *geom = DRW_cache_gpencil_strokes_get(iter.ob, iter.cfra);
    /* Start one vert before to have gl_InstanceID > 0 (see shader). */
    int vfirst = gps->runtime.stroke_start - 1;
    /* Include "potential" cyclic vertex and start adj vertex (see shader). */
    int vcount = gps->totpoints + 1 + 1;
    gpencil_drawcall_add(iter, geom, material, vfirst, vcount, true);
  }
}

void Instance::gpencil_sync(Object *ob, ObjectHandle &ob_handle)
{
  gpIterData iter(*this, ob);

  BKE_gpencil_visible_stroke_iter((bGPdata *)ob->data, nullptr, gpencil_stroke_sync, &iter);

  gpencil_drawcall_flush(iter);

  /* TODO(fclem) Gpencil velocity. */
  // shading_passes.velocity.gpencil_add(ob, ob_handle);

  bool is_caster = true;      /* TODO material.shadow.shgrp. */
  bool is_alpha_blend = true; /* TODO material.is_alpha_blend. */
  shadows.sync_object(ob, ob_handle, is_caster, is_alpha_blend);
}

}  // namespace blender::eevee
