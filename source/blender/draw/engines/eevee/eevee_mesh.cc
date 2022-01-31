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

#include "eevee_instance.hh"

namespace blender::eevee {

void Instance::mesh_sync(Object *ob, ObjectHandle &ob_handle)
{
  MaterialArray &material_array = materials.material_array_get(ob);

  GPUBatch **mat_geom = DRW_cache_object_surface_material_get(
      ob, material_array.gpu_materials.data(), material_array.gpu_materials.size());

  if (mat_geom == nullptr) {
    return;
  }

  bool is_shadow_caster = false;
  bool is_alpha_blend = false;
  for (auto i : material_array.gpu_materials.index_range()) {
    GPUBatch *geom = mat_geom[i];
    if (geom == nullptr) {
      continue;
    }
    Material *material = material_array.materials[i];
    shgroup_geometry_call(material->shading.shgrp, ob, geom);
    shgroup_geometry_call(material->prepass.shgrp, ob, geom);
    shgroup_geometry_call(material->shadow.shgrp, ob, geom);

    is_shadow_caster = is_shadow_caster || material->shadow.shgrp != nullptr;
    is_alpha_blend = is_alpha_blend || material->is_alpha_blend_transparent;
  }
  shading_passes.velocity.mesh_add(ob, ob_handle);

  shadows.sync_object(ob, ob_handle, is_shadow_caster, is_alpha_blend);
}

}  // namespace blender::eevee
