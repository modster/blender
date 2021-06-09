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
 */

#include "BLI_utildefines.h"

#include "BKE_context.h"
#include "BKE_modifier.h"

#include "DNA_screen_types.h"

#include "RNA_access.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "MOD_ui_common.h"

static Mesh *modifyMesh(ModifierData *UNUSED(md),
                        const ModifierEvalContext *UNUSED(ctx),
                        Mesh *mesh)
{
  return mesh;
}

static bool dependsOnTime(ModifierData *UNUSED(md))
{
  return true;
}

static void panel_draw(const bContext *UNUSED(C), Panel *panel)
{
  uiLayout *layout = panel->layout;

  PointerRNA ob_ptr;
  PointerRNA *ptr = modifier_panel_get_property_pointers(panel, &ob_ptr);

  uiLayoutSetPropSep(layout, true);

  modifier_panel_end(layout, ptr);
}

static void panelRegister(ARegionType *region_type)
{
  modifier_panel_register(region_type, eModifierType_AdaptiveRemesh, panel_draw);
}

ModifierTypeInfo modifierType_AdaptiveRemesh = {
    /* name */ "AdaptiveRemesh",
    /* structName */ "AdaptiveRemeshModifierData",
    /* structSize */ sizeof(AdaptiveRemeshModifierData),
    /* srna */ &RNA_AdaptiveRemeshModifier,
    /* type */ eModifierTypeType_Nonconstructive,
    /* flags */ eModifierTypeFlag_AcceptsMesh,
    /* icon */ ICON_MOD_CLOTH, /* TODO(ish): Use correct icon. */

    /* copyData */ nullptr,

    /* deformVerts */ nullptr,
    /* deformMatrices */ nullptr,
    /* deformVertsEM */ nullptr,
    /* deformMatricesEM */ nullptr,
    /* modifyMesh */ modifyMesh,
    /* modifyHair */ nullptr,
    /* modifyGeometrySet */ nullptr,

    /* initData */ nullptr,
    /* requiredDataMask */ nullptr,
    /* freeData */ nullptr,
    /* isDisabled */ nullptr,
    /* updateDepsgraph */ NULL,
    /* dependsOnTime */ dependsOnTime,
    /* dependsOnNormals */ nullptr,
    /* foreachIDLink */ nullptr,
    /* foreachTexLink */ nullptr,
    /* freeRuntimeData */ nullptr,
    /* panelRegister */ panelRegister,
    /* blendWrite */ nullptr,
    /* blendRead */ nullptr,
};
