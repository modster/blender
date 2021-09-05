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

#include "BLI_assert.h"
#include "BLI_string.h"
#include "BLI_utildefines.h"

#include "BKE_cloth_remesh.hh"
#include "BKE_context.h"
#include "BKE_modifier.h"

#include "DNA_defaults.h"
#include "DNA_modifier_types.h"
#include "DNA_screen_types.h"

#include "RNA_access.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "MOD_ui_common.h"

using namespace blender::bke;

static internal::FilenameGen split_edge_name_gen("/tmp/adaptive_cloth/split_edge", ".mesh");
static internal::FilenameGen collapse_edge_name_gen("/tmp/adaptive_cloth/collapse_edge", ".mesh");
static internal::FilenameGen flip_edge_name_gen("/tmp/adaptive_cloth/flip_edge", ".mesh");

static void initData(ModifierData *md)
{
  AdaptiveRemeshModifierData *armd = reinterpret_cast<AdaptiveRemeshModifierData *>(md);
  BLI_assert(MEMCMP_STRUCT_AFTER_IS_ZERO(armd, modifier));

  MEMCPY_STRUCT_AFTER(armd, DNA_struct_default_get(AdaptiveRemeshModifierData), modifier);
}

static Mesh *modifyMesh(ModifierData *md, const ModifierEvalContext *UNUSED(ctx), Mesh *mesh)
{
  AdaptiveRemeshModifierData *armd = (AdaptiveRemeshModifierData *)md;

  auto mode = armd->mode;

  if (mode == ADAPTIVE_REMESH_STATIC_REMESHING || mode == ADAPTIVE_REMESH_DYNAMIC_REMESHING) {
    TempEmptyAdaptiveRemeshParams params;
    params.edge_length_min = armd->edge_length_min;
    params.edge_length_max = armd->edge_length_max;
    params.aspect_ratio_min = armd->aspect_ratio_min;
    params.change_in_vertex_normal_max = armd->change_in_vertex_normal_max;
    params.flags = 0;
    if (armd->flag & ADAPTIVE_REMESH_SEWING) {
      params.flags |= ADAPTIVE_REMESH_PARAMS_SEWING;
    }
    if (armd->flag & ADAPTIVE_REMESH_FORCE_SPLIT_FOR_SEWING) {
      params.flags |= ADAPTIVE_REMESH_PARAMS_FORCE_SPLIT_FOR_SEWING;
    }
    if (mode == ADAPTIVE_REMESH_STATIC_REMESHING) {
      params.type = ADAPTIVE_REMESH_PARAMS_STATIC_REMESH;
    }
    else if (mode == ADAPTIVE_REMESH_DYNAMIC_REMESHING) {
      params.type = ADAPTIVE_REMESH_PARAMS_DYNAMIC_REMESH;
    }
    else {
      BLI_assert_unreachable();
    }

    return __temp_empty_adaptive_remesh(params, mesh);
  }

  auto edge_i = armd->edge_index;
  auto across_seams = armd->flag & ADAPTIVE_REMESH_ACROSS_SEAMS;
  auto verts_swapped = armd->flag & ADAPTIVE_REMESH_VERTS_SWAPPED;

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
    char filename_pre_suffix_c[16];
    BLI_snprintf(filename_pre_suffix_c, 16, "%03d", edge_i);
    std::string filename_pre_suffix(filename_pre_suffix_c);

    auto edge_index = op_edge_index.value();
    std::cout << "edge_index: " << edge_index << " edge_i: " << armd->edge_index
              << " across_seams: " << across_seams << " mode: " << mode << std::endl;
    bool is_on_boundary = internal_mesh.is_edge_on_boundary(edge_index);
    std::cout << "is_on_boundary: " << is_on_boundary << std::endl;
    auto flippable = internal_mesh.is_edge_flippable(edge_index, across_seams);
    std::cout << "flippable: " << flippable << std::endl;
    auto collapseable = internal_mesh.is_edge_collapseable(
        edge_index, verts_swapped, across_seams);
    std::cout << "collapseable: " << collapseable << std::endl;
    if (mode == ADAPTIVE_REMESH_SPLIT_EDGE) {
      auto pre_split_msgpack = internal_mesh.serialize();
      auto pre_split_filename = split_edge_name_gen.get_curr(filename_pre_suffix + "_pre");

      internal_mesh.split_edge_triangulate(edge_index, across_seams, true);

      auto post_split_msgpack = internal_mesh.serialize();
      auto post_split_filename = split_edge_name_gen.get_curr(filename_pre_suffix + "_post");
      /* split_edge_name_gen.gen_next(); */

      internal::dump_file(pre_split_filename, pre_split_msgpack);
      internal::dump_file(post_split_filename, post_split_msgpack);
    }
    else if (mode == ADAPTIVE_REMESH_COLLAPSE_EDGE) {
      if (collapseable) {
        auto pre_collapse_msgpack = internal_mesh.serialize();
        auto pre_collapse_filename = collapse_edge_name_gen.get_curr(filename_pre_suffix + "_pre");

        internal_mesh.collapse_edge_triangulate(edge_index, verts_swapped, across_seams);

        auto post_collapse_msgpack = internal_mesh.serialize();
        auto post_collapse_filename = collapse_edge_name_gen.get_curr(filename_pre_suffix +
                                                                      "_post");
        /* collapse_edge_name_gen.gen_next(); */

        internal::dump_file(pre_collapse_filename, pre_collapse_msgpack);
        internal::dump_file(post_collapse_filename, post_collapse_msgpack);
      }
    }
    else if (mode == ADAPTIVE_REMESH_FLIP_EDGE) {
      if (flippable) {
        auto pre_flip_msgpack = internal_mesh.serialize();
        auto pre_flip_filename = flip_edge_name_gen.get_curr(filename_pre_suffix + "_pre");

        internal_mesh.flip_edge_triangulate(edge_index, across_seams);

        auto post_flip_msgpack = internal_mesh.serialize();
        auto post_flip_filename = flip_edge_name_gen.get_curr(filename_pre_suffix + "_post");
        /* flip_edge_name_gen.gen_next(); */

        internal::dump_file(pre_flip_filename, pre_flip_msgpack);
        internal::dump_file(post_flip_filename, post_flip_msgpack);
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
  AdaptiveRemeshModifierData *armd = static_cast<AdaptiveRemeshModifierData *>(ptr->data);

  uiLayoutSetPropSep(layout, true);

  uiItemR(layout, ptr, "mode", 0, nullptr, ICON_NONE);
  if (armd->mode == ADAPTIVE_REMESH_SPLIT_EDGE || armd->mode == ADAPTIVE_REMESH_COLLAPSE_EDGE ||
      armd->mode == ADAPTIVE_REMESH_FLIP_EDGE) {
    uiItemR(layout, ptr, "edge_index", 0, nullptr, ICON_NONE);
    uiItemR(layout, ptr, "use_across_seams", 0, nullptr, ICON_NONE);
    uiItemR(layout, ptr, "is_verts_swapped", 0, nullptr, ICON_NONE);
  }
  else if (armd->mode == ADAPTIVE_REMESH_STATIC_REMESHING) {
    uiItemR(layout, ptr, "edge_length_min", 0, nullptr, ICON_NONE);
    uiItemR(layout, ptr, "aspect_ratio_min", 0, nullptr, ICON_NONE);
    uiItemR(layout, ptr, "enable_sewing", 0, nullptr, ICON_NONE);
    uiItemR(layout, ptr, "force_split_for_sewing", 0, nullptr, ICON_NONE);
  }
  else if (armd->mode == ADAPTIVE_REMESH_DYNAMIC_REMESHING) {
    uiItemR(layout, ptr, "edge_length_min", 0, nullptr, ICON_NONE);
    uiItemR(layout, ptr, "edge_length_max", 0, nullptr, ICON_NONE);
    uiItemR(layout, ptr, "aspect_ratio_min", 0, nullptr, ICON_NONE);
    uiItemR(layout, ptr, "change_in_vertex_normal_max", 0, nullptr, ICON_NONE);
    uiItemR(layout, ptr, "enable_sewing", 0, nullptr, ICON_NONE);
    uiItemR(layout, ptr, "force_split_for_sewing", 0, nullptr, ICON_NONE);
  }
  else {
    BLI_assert_unreachable();
  }

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

    /* initData */ initData,
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
