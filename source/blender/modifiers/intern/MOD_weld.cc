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
 * The Original Code is Copyright (C) 2005 by the Blender Foundation.
 * All rights reserved.
 */

/** \file
 * \ingroup modifiers
 *
 * Weld modifier: Remove doubles.
 */

/* TODOs:
 * - Review weight and vertex color interpolation.;
 */

//#define USE_WELD_DEBUG
//#define USE_WELD_NORMALS
//#define USE_BVHTREEKDOP

#include "MEM_guardedalloc.h"

#include "BLI_math.h"
#include "BLI_utildefines.h"

#include "BLT_translation.h"

#include "DNA_defaults.h"
#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"
#include "DNA_modifier_types.h"
#include "DNA_object_types.h"
#include "DNA_screen_types.h"

#ifdef USE_BVHTREEKDOP
#  include "BKE_bvhutils.h"
#endif

#include "BKE_context.h"
#include "BKE_deform.h"
#include "BKE_modifier.h"
#include "BKE_screen.h"

#include "GEO_mesh_merge_by_distance.hh"

#include "UI_interface.h"
#include "UI_resources.h"

#include "RNA_access.h"

#include "MOD_ui_common.h"

static Mesh *modifyMesh(ModifierData *md, const ModifierEvalContext *UNUSED(ctx), Mesh *mesh)
{
  WeldModifierData *wmd = (WeldModifierData *)md;

  uint totvert = mesh->totvert;
  bool *mask = (bool *)MEM_malloc_arrayN(totvert, sizeof(*mask), __func__);
  const int defgrp_index = BKE_id_defgroup_name_index(&mesh->id, wmd->defgrp_name);
  if (defgrp_index != -1) {
    MDeformVert *dvert, *dv;
    dvert = (MDeformVert *)CustomData_get_layer(&mesh->vdata, CD_MDEFORMVERT);
    if (dvert) {
      const bool invert_vgroup = (wmd->flag & MOD_WELD_INVERT_VGROUP) != 0;
      dv = &dvert[0];
      for (uint i = 0; i < totvert; i++, dv++) {
        const bool found = BKE_defvert_find_weight(dv, defgrp_index) > 0.0f;
        mask[i] = found != invert_vgroup;
      }
    }
  }
  else {
    for (int i = 0; i < totvert; i++) {
      mask[i] = true;
    }
  }

  Mesh *result = blender::geometry::GEO_mesh_merge_by_distance(
      mesh, mask, wmd->merge_dist, wmd->mode);
  MEM_freeN(mask);

  return result;
}

static void initData(ModifierData *md)
{
  WeldModifierData *wmd = (WeldModifierData *)md;

  BLI_assert(MEMCMP_STRUCT_AFTER_IS_ZERO(wmd, modifier));

  MEMCPY_STRUCT_AFTER(wmd, DNA_struct_default_get(WeldModifierData), modifier);
}

static void requiredDataMask(Object *UNUSED(ob),
                             ModifierData *md,
                             CustomData_MeshMasks *r_cddata_masks)
{
  WeldModifierData *wmd = (WeldModifierData *)md;

  /* Ask for vertexgroups if we need them. */
  if (wmd->defgrp_name[0] != '\0') {
    r_cddata_masks->vmask |= CD_MASK_MDEFORMVERT;
  }
}

static void panel_draw(const bContext *UNUSED(C), Panel *panel)
{
  uiLayout *layout = panel->layout;

  PointerRNA ob_ptr;
  PointerRNA *ptr = modifier_panel_get_property_pointers(panel, &ob_ptr);
  int weld_mode = RNA_enum_get(ptr, "mode");

  uiLayoutSetPropSep(layout, true);

  uiItemR(layout, ptr, "mode", 0, NULL, ICON_NONE);
  uiItemR(layout, ptr, "merge_threshold", 0, IFACE_("Distance"), ICON_NONE);
  if (weld_mode == blender::geometry::WELD_MODE_CONNECTED) {
    uiItemR(layout, ptr, "loose_edges", 0, NULL, ICON_NONE);
  }
  modifier_vgroup_ui(layout, ptr, &ob_ptr, "vertex_group", "invert_vertex_group", NULL);

  modifier_panel_end(layout, ptr);
}

static void panelRegister(ARegionType *region_type)
{
  modifier_panel_register(region_type, eModifierType_Weld, panel_draw);
}

ModifierTypeInfo modifierType_Weld = {
    /* name */ "Weld",
    /* structName */ "WeldModifierData",
    /* structSize */ sizeof(WeldModifierData),
    /* srna */ &RNA_WeldModifier,
    /* type */ eModifierTypeType_Constructive,
    /* flags */
    (ModifierTypeFlag)(eModifierTypeFlag_AcceptsMesh | eModifierTypeFlag_SupportsMapping |
                       eModifierTypeFlag_SupportsEditmode | eModifierTypeFlag_EnableInEditmode |
                       eModifierTypeFlag_AcceptsCVs),
    /* icon */ ICON_AUTOMERGE_OFF, /* TODO: Use correct icon. */

    /* copyData */ BKE_modifier_copydata_generic,

    /* deformVerts */ NULL,
    /* deformMatrices */ NULL,
    /* deformVertsEM */ NULL,
    /* deformMatricesEM */ NULL,
    /* modifyMesh */ modifyMesh,
    /* modifyHair */ NULL,
    /* modifyGeometrySet */ NULL,

    /* initData */ initData,
    /* requiredDataMask */ requiredDataMask,
    /* freeData */ NULL,
    /* isDisabled */ NULL,
    /* updateDepsgraph */ NULL,
    /* dependsOnTime */ NULL,
    /* dependsOnNormals */ NULL,
    /* foreachIDLink */ NULL,
    /* foreachTexLink */ NULL,
    /* freeRuntimeData */ NULL,
    /* panelRegister */ panelRegister,
    /* blendWrite */ NULL,
    /* blendRead */ NULL,
};

/** \} */
