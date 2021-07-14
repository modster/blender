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

#include "BKE_cloth_remesh.hh"
#include "BKE_context.h"
#include "BKE_modifier.h"

#include "DNA_modifier_types.h"
#include "DNA_screen_types.h"

#include "RNA_access.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "MOD_ui_common.h"

using namespace blender::bke;

static Mesh *modifyMesh(ModifierData *md, const ModifierEvalContext *UNUSED(ctx), Mesh *mesh)
{
  AdaptiveRemeshModifierData *armd = (AdaptiveRemeshModifierData *)md;

  auto edge_i = armd->edge_index;
  auto across_seams = armd->flag & ADAPTIVE_REMESH_ACROSS_SEAMS;
  auto verts_swapped = armd->flag & ADAPTIVE_REMESH_VERTS_SWAPPED;
  auto mode = armd->mode;

  internal::MeshIO reader;
  reader.read(mesh);

  internal::Mesh<internal::EmptyExtraData,
                 internal::EmptyExtraData,
                 internal::EmptyExtraData,
                 internal::EmptyExtraData>
      internal_mesh;
  internal_mesh.read(reader);

  auto op_edge_index = internal_mesh.get_edges().get_no_gen_index(edge_i);
  if (op_edge_index) {
    auto edge_index = op_edge_index.value();
    std::cout << "edge_index: " << edge_index << " edge_i: " << armd->edge_index
              << " across_seams: " << across_seams << " mode: " << mode << std::endl;
    bool is_on_boundary = internal_mesh.is_edge_on_boundary(edge_index);
    std::cout << "is_on_boundary: " << is_on_boundary << std::endl;
    if (mode == ADAPTIVE_REMESH_SPLIT_EDGE) {
      internal_mesh.split_edge_triangulate(edge_index, across_seams);
    }
    else if (mode == ADAPTIVE_REMESH_COLLAPSE_EDGE) {
      internal_mesh.collapse_edge_triangulate(edge_index, verts_swapped, across_seams);
    }
    else if (mode == ADAPTIVE_REMESH_FLIP_EDGE) {
      auto flippable = internal_mesh.is_edge_flippable(edge_index, across_seams);
      std::cout << "flippable: " << flippable << std::endl;
      if (flippable) {
        internal_mesh.flip_edge_triangulate(edge_index, across_seams);
      }
    }
  }

  internal::MeshIO writer = internal_mesh.write();
  auto *mesh_result = writer.write();
  return mesh_result;
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

  uiItemR(layout, ptr, "mode", 0, nullptr, ICON_NONE);
  uiItemR(layout, ptr, "edge_index", 0, nullptr, ICON_NONE);
  uiItemR(layout, ptr, "use_across_seams", 0, nullptr, ICON_NONE);
  uiItemR(layout, ptr, "is_verts_swapped", 0, nullptr, ICON_NONE);

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

    /* copyData */ BKE_modifier_copydata_generic,

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
    /* updateDepsgraph */ nullptr,
    /* dependsOnTime */ dependsOnTime,
    /* dependsOnNormals */ nullptr,
    /* foreachIDLink */ nullptr,
    /* foreachTexLink */ nullptr,
    /* freeRuntimeData */ nullptr,
    /* panelRegister */ panelRegister,
    /* blendWrite */ nullptr,
    /* blendRead */ nullptr,
};
