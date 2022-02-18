/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2011 by Nicholas Bishop. */

/** \file
 * \ingroup modifiers
 */

#include "BLI_utildefines.h"

#include "DNA_defaults.h"
#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"
#include "DNA_modifier_types.h"
#include "DNA_object_types.h"
#include "DNA_screen_types.h"

#include "BKE_context.h"
#include "BKE_mesh.h"
#include "BKE_mesh_remesh_voxel.h"
#include "BKE_screen.h"

#include "GEO_mesh_remesh_blocks.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "RNA_access.h"

#include "MOD_modifiertypes.h"
#include "MOD_ui_common.h"

#include <string.h>

static void initData(ModifierData *md)
{
  RemeshModifierData *rmd = (RemeshModifierData *)md;

  BLI_assert(MEMCMP_STRUCT_AFTER_IS_ZERO(rmd, modifier));

  MEMCPY_STRUCT_AFTER(rmd, DNA_struct_default_get(RemeshModifierData), modifier);
}

static Mesh *modifyMesh(ModifierData *md, const ModifierEvalContext *UNUSED(ctx), Mesh *mesh)
{
  RemeshModifierData *rmd;
  Mesh *result;

  rmd = (RemeshModifierData *)md;

  if (rmd->mode == MOD_REMESH_VOXEL) {
    /* OpenVDB modes. */
    if (rmd->voxel_size == 0.0f) {
      return NULL;
    }
    result = BKE_mesh_remesh_voxel(mesh, rmd->voxel_size, rmd->adaptivity, 0.0f);
    if (result == NULL) {
      return NULL;
    }
  }
  else {
    result = GEO_mesh_remesh_blocks(
        mesh, rmd->flag, rmd->mode, rmd->threshold, rmd->hermite_num, rmd->scale, rmd->depth);
  }

  if (rmd->flag & MOD_REMESH_SMOOTH_SHADING) {
    MPoly *mpoly = result->mpoly;
    int i, totpoly = result->totpoly;

    /* Apply smooth shading to output faces */
    for (i = 0; i < totpoly; i++) {
      mpoly[i].flag |= ME_SMOOTH;
    }
  }

  BKE_mesh_copy_parameters_for_eval(result, mesh);
  BKE_mesh_calc_edges(result, true, false);
  BKE_mesh_normals_tag_dirty(result);
  return result;
}

static void panel_draw(const bContext *UNUSED(C), Panel *panel)
{
  uiLayout *layout = panel->layout;
#ifdef WITH_MOD_REMESH
  uiLayout *row, *col;

  PointerRNA ob_ptr;
  PointerRNA *ptr = modifier_panel_get_property_pointers(panel, &ob_ptr);

  int mode = RNA_enum_get(ptr, "mode");

  uiItemR(layout, ptr, "mode", UI_ITEM_R_EXPAND, NULL, ICON_NONE);

  uiLayoutSetPropSep(layout, true);

  col = uiLayoutColumn(layout, false);
  if (mode == MOD_REMESH_VOXEL) {
    uiItemR(col, ptr, "voxel_size", 0, NULL, ICON_NONE);
    uiItemR(col, ptr, "adaptivity", 0, NULL, ICON_NONE);
  }
  else {
    uiItemR(col, ptr, "octree_depth", 0, NULL, ICON_NONE);
    uiItemR(col, ptr, "scale", 0, NULL, ICON_NONE);

    if (mode == MOD_REMESH_SHARP_FEATURES) {
      uiItemR(col, ptr, "sharpness", 0, NULL, ICON_NONE);
    }

    uiItemR(layout, ptr, "use_remove_disconnected", 0, NULL, ICON_NONE);
    row = uiLayoutRow(layout, false);
    uiLayoutSetActive(row, RNA_boolean_get(ptr, "use_remove_disconnected"));
    uiItemR(layout, ptr, "threshold", 0, NULL, ICON_NONE);
  }
  uiItemR(layout, ptr, "use_smooth_shade", 0, NULL, ICON_NONE);

  modifier_panel_end(layout, ptr);

#else  /* WITH_MOD_REMESH */
  uiItemL(layout, TIP_("Built without Remesh modifier"), ICON_NONE);
#endif /* WITH_MOD_REMESH */
}

static void panelRegister(ARegionType *region_type)
{
  modifier_panel_register(region_type, eModifierType_Remesh, panel_draw);
}

ModifierTypeInfo modifierType_Remesh = {
    /* name */ "Remesh",
    /* structName */ "RemeshModifierData",
    /* structSize */ sizeof(RemeshModifierData),
    /* srna */ &RNA_RemeshModifier,
    /* type */ eModifierTypeType_Nonconstructive,
    /* flags */ eModifierTypeFlag_AcceptsMesh | eModifierTypeFlag_AcceptsCVs |
        eModifierTypeFlag_SupportsEditmode,
    /* icon */ ICON_MOD_REMESH,

    /* copyData */ BKE_modifier_copydata_generic,

    /* deformVerts */ NULL,
    /* deformMatrices */ NULL,
    /* deformVertsEM */ NULL,
    /* deformMatricesEM */ NULL,
    /* modifyMesh */ modifyMesh,
    /* modifyGeometrySet */ NULL,

    /* initData */ initData,
    /* requiredDataMask */ NULL,
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
