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

#include "DNA_hair_types.h"
#include "DNA_modifier_types.h"
#include "DNA_particle_types.h"

#include "eevee_instance.hh"

namespace blender::eevee {

static void shgroup_hair_call(MaterialPass &matpass,
                              Object *ob,
                              ParticleSystem *part_sys = nullptr,
                              ModifierData *modifier_data = nullptr)
{
  if (matpass.shgrp == nullptr) {
    return;
  }
  DRW_shgroup_hair_create_sub(ob, part_sys, modifier_data, matpass.shgrp, matpass.gpumat);
}

void Instance::hair_sync(Object *ob, ObjectHandle &ob_handle, ModifierData *modifier_data)
{
  int mat_nr = HAIR_MATERIAL_NR;

  ParticleSystem *part_sys = nullptr;
  if (modifier_data != nullptr) {
    part_sys = reinterpret_cast<ParticleSystemModifierData *>(modifier_data)->psys;
    if (!DRW_object_is_visible_psys_in_active_context(ob, part_sys)) {
      return;
    }
    ParticleSettings *part_settings = part_sys->part;
    const int draw_as = (part_settings->draw_as == PART_DRAW_REND) ? part_settings->ren_as :
                                                                     part_settings->draw_as;
    if (draw_as != PART_DRAW_PATH) {
      return;
    }
    mat_nr = part_settings->omat;
  }

  Material &material = materials.material_get(ob, mat_nr - 1, MAT_GEOM_HAIR);

  shgroup_hair_call(material.shading, ob, part_sys, modifier_data);
  shgroup_hair_call(material.prepass, ob, part_sys, modifier_data);
  shgroup_hair_call(material.shadow, ob, part_sys, modifier_data);
  /* TODO(fclem) Hair velocity. */
  // shading_passes.velocity.gpencil_add(ob, ob_handle);

  bool is_caster = material.shadow.shgrp != nullptr;
  bool is_alpha_blend = material.is_alpha_blend_transparent;
  shadows.sync_object(ob, ob_handle, is_caster, is_alpha_blend);
}

}  // namespace blender::eevee
