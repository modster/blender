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

#include "stdio.h"
#include <BLI_string.h>
#include <MEM_guardedalloc.h>
#include <string.h>

#include "BLI_utildefines.h"

#include "BKE_lattice.h"
#include "BLT_translation.h"

#include "DNA_defaults.h"
#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"
#include "DNA_object_types.h"
#include "DNA_screen_types.h"

#include "BKE_context.h"
#include "BKE_deform.h"
#include "BKE_particle.h"
#include "BKE_screen.h"
#include "BKE_solidifiy.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "RNA_access.h"

#include "MOD_modifiertypes.h"
#include "MOD_ui_common.h"

#include "MOD_solidify_util.h"

static bool dependsOnNormals(ModifierData *md)
{
  const SolidifyModifierData *smd = (SolidifyModifierData *)md;
  /* even when we calculate our own normals,
   * the vertex normals are used as a fallback
   * if manifold is enabled vertex normals are not used */
  return smd->mode == MOD_SOLIDIFY_MODE_EXTRUDE;
}

static void initData(ModifierData *md)
{
  SolidifyModifierData *smd = (SolidifyModifierData *)md;

  BLI_assert(MEMCMP_STRUCT_AFTER_IS_ZERO(smd, modifier));

  MEMCPY_STRUCT_AFTER(smd, DNA_struct_default_get(SolidifyModifierData), modifier);
}

#ifdef __GNUC__
#  pragma GCC diagnostic error "-Wsign-conversion"
#endif

static void requiredDataMask(Object *UNUSED(ob),
                             ModifierData *md,
                             CustomData_MeshMasks *r_cddata_masks)
{
  SolidifyModifierData *smd = (SolidifyModifierData *)md;

  /* ask for vertexgroups if we need them */
  if (smd->defgrp_name[0] != '\0' || smd->shell_defgrp_name[0] != '\0' ||
      smd->rim_defgrp_name[0] != '\0') {
    r_cddata_masks->vmask |= CD_MASK_MDEFORMVERT;
  }
}

static float *get_distance_factor(Mesh *mesh, Object *ob, const char *name, bool invert)
{
  int defgrp_index = BKE_object_defgroup_name_index(ob, name);
  MDeformVert *dvert = mesh->dvert;

  float *selection = MEM_callocN(sizeof(float) * (unsigned long)mesh->totvert, __func__);

  if (defgrp_index != -1) {
    if (ob->type == OB_LATTICE) {
      dvert = BKE_lattice_deform_verts_get(ob);
    }
    else if (mesh) {
      dvert = mesh->dvert;
      for (int i = 0; i < mesh->totvert; i++) {
        MDeformVert *dv = &dvert[i];
        printf("sel: %i\n", defgrp_index);
        selection[i] = BKE_defvert_find_weight(dv, defgrp_index);
      }
    }
  }
  else {
    for (int i = 0; i < mesh->totvert; i++) {
      selection[i] = 1.0f;
    }
  }

  if(invert){
    for (int i = 0; i < mesh->totvert; i++) {
      selection[i] = 1.0f - selection[i];
    }
  }

  return selection;
}

static const SolidifyData solidify_data_from_modifier_data(ModifierData *md,
                                                           const ModifierEvalContext *ctx)
{
  const SolidifyModifierData *smd = (SolidifyModifierData *)md;
  SolidifyData solidify_data = {
      ctx->object,
      "",
      "",
      "",
      smd->offset,
      smd->offset_fac,
      smd->offset_fac_vg,
      smd->offset_clamp,
      smd->mode,
      smd->nonmanifold_offset_mode,
      smd->nonmanifold_boundary_mode,
      smd->crease_inner,
      smd->crease_outer,
      smd->crease_rim,
      smd->flag,
      smd->mat_ofs,
      smd->mat_ofs_rim,
      smd->mode == MOD_SOLIDIFY_MODE_EXTRUDE ? 0.01f : smd->merge_tolerance,
      smd->bevel_convex,
      NULL,
  };

  BLI_strncpy(solidify_data.defgrp_name, smd->defgrp_name, MAX_NAME);
  BLI_strncpy(solidify_data.shell_defgrp_name, smd->shell_defgrp_name, MAX_NAME);
  BLI_strncpy(solidify_data.rim_defgrp_name, smd->rim_defgrp_name, MAX_NAME);

  if (!(smd->flag & MOD_SOLIDIFY_NOSHELL)) {
    solidify_data.flag |= MOD_SOLIDIFY_SHELL;
  }

  return solidify_data;
}

static Mesh *MOD_solidify_nonmanifold(ModifierData *md,
                                      const ModifierEvalContext *ctx,
                                      Mesh *mesh,
                                      const SolidifyModifierData *smd)
{
  SolidifyData solidify_data = solidify_data_from_modifier_data(md, ctx);

  const bool defgrp_invert = (solidify_data.flag & MOD_SOLIDIFY_VGROUP_INV) != 0;
  solidify_data.distance = get_distance_factor(
      mesh, ctx->object, smd->defgrp_name, defgrp_invert);

  bool *shell_verts = NULL;
  bool *rim_verts = NULL;
  bool *shell_faces = NULL;
  bool *rim_faces = NULL;

  Mesh *output_mesh = solidify_nonmanifold(&solidify_data, mesh, &shell_verts, &rim_verts, &shell_faces, &rim_faces);

  const int shell_defgrp_index = BKE_object_defgroup_name_index(ctx->object,
                                                                smd->shell_defgrp_name);
  const int rim_defgrp_index = BKE_object_defgroup_name_index(ctx->object,
                                                              smd->rim_defgrp_name);

  MDeformVert *dvert;
  if (shell_defgrp_index != -1 || rim_defgrp_index != -1) {
    dvert = CustomData_duplicate_referenced_layer(
        &output_mesh->vdata, CD_MDEFORMVERT, output_mesh->totvert);
    /* If no vertices were ever added to an object's vgroup, dvert might be NULL. */
    if (dvert == NULL) {
      /* Add a valid data layer! */
      dvert = CustomData_add_layer(
          &output_mesh->vdata, CD_MDEFORMVERT, CD_CALLOC, NULL, output_mesh->totvert);
    }
    output_mesh->dvert = dvert;
    if ((solidify_data.flag & MOD_SOLIDIFY_SHELL) && shell_defgrp_index != -1) {
      for (int i = 0; i < output_mesh->totvert; i++) {
        BKE_defvert_ensure_index(&output_mesh->dvert[i], shell_defgrp_index)->weight =
            shell_verts[i];
      }
    }
    if ((solidify_data.flag & MOD_SOLIDIFY_RIM) && rim_defgrp_index != -1) {
      for (int i = 0; i < output_mesh->totvert; i++) {
        BKE_defvert_ensure_index(&output_mesh->dvert[i], rim_defgrp_index)->weight =
            rim_verts[i];
      }
    }
  }

  /* Only use material offsets if we have 2 or more materials. */
  const short mat_nrs = solidify_data.object->totcol > 1 ? solidify_data.object->totcol : 1;
  const short mat_nr_max = mat_nrs - 1;
  const short mat_ofs = mat_nrs > 1 ? solidify_data.mat_ofs : 0;
  const short mat_ofs_rim = mat_nrs > 1 ? solidify_data.mat_ofs_rim : 0;

  short most_mat_nr = 0;
  uint most_mat_nr_count = 0;
  for(int mat_nr = 0; mat_nr < mat_nrs; mat_nr++){
    uint count = 0;
    for(int i = 0; i < mesh->totpoly; i++){
      if(mesh->mpoly[i].mat_nr == mat_nr){
        count++;
      }
    }
    if(count > most_mat_nr_count){
      most_mat_nr = mat_nr;
    }
  }

  for(int i = 0; i < output_mesh->totpoly; i++){
    output_mesh->mpoly[i].mat_nr = most_mat_nr;
    if(mat_ofs > 0 && shell_faces && shell_faces[i]){
      output_mesh->mpoly[i].mat_nr += mat_ofs;
      CLAMP(output_mesh->mpoly[i].mat_nr, 0, mat_nr_max);
    }
    else if(mat_ofs_rim > 0 && rim_faces && rim_faces[i]){
      output_mesh->mpoly[i].mat_nr += mat_ofs_rim;
      CLAMP(output_mesh->mpoly[i].mat_nr, 0, mat_nr_max);
    }
  }

  MEM_freeN(solidify_data.distance);
  MEM_freeN(shell_verts);
  MEM_freeN(rim_verts);
  MEM_freeN(shell_faces);
  MEM_freeN(rim_faces);
  return output_mesh;
}

static Mesh *modifyMesh(ModifierData *md, const ModifierEvalContext *ctx, Mesh *mesh)
{
  const SolidifyModifierData *smd = (SolidifyModifierData *)md;

  switch (smd->mode) {
    case MOD_SOLIDIFY_MODE_EXTRUDE:
      return MOD_solidify_extrude_modifyMesh(md, ctx, mesh);
    case MOD_SOLIDIFY_MODE_NONMANIFOLD: {
      return MOD_solidify_nonmanifold(md, ctx, mesh, smd);
    }
    default:
      BLI_assert(0);
  }
  return mesh;
}

static void panel_draw(const bContext *UNUSED(C), Panel *panel)
{
  uiLayout *sub, *row, *col;
  uiLayout *layout = panel->layout;

  PointerRNA ob_ptr;
  PointerRNA *ptr = modifier_panel_get_property_pointers(panel, &ob_ptr);

  int solidify_mode = RNA_enum_get(ptr, "solidify_mode");
  bool has_vertex_group = RNA_string_length(ptr, "vertex_group") != 0;

  uiLayoutSetPropSep(layout, true);

  uiItemR(layout, ptr, "solidify_mode", 0, NULL, ICON_NONE);

  if (solidify_mode == MOD_SOLIDIFY_MODE_NONMANIFOLD) {
    uiItemR(layout, ptr, "nonmanifold_thickness_mode", 0, IFACE_("Thickness Mode"), ICON_NONE);
    uiItemR(layout, ptr, "nonmanifold_boundary_mode", 0, IFACE_("Boundary"), ICON_NONE);
  }

  uiItemR(layout, ptr, "thickness", 0, NULL, ICON_NONE);
  uiItemR(layout, ptr, "offset", 0, NULL, ICON_NONE);

  if (solidify_mode == MOD_SOLIDIFY_MODE_NONMANIFOLD) {
    uiItemR(layout, ptr, "nonmanifold_merge_threshold", 0, NULL, ICON_NONE);
  }
  else {
    uiItemR(layout, ptr, "use_even_offset", 0, NULL, ICON_NONE);
  }

  col = uiLayoutColumnWithHeading(layout, false, IFACE_("Rim"));
  uiItemR(col, ptr, "use_rim", 0, IFACE_("Fill"), ICON_NONE);
  sub = uiLayoutColumn(col, false);
  uiLayoutSetActive(sub, RNA_boolean_get(ptr, "use_rim"));
  uiItemR(sub, ptr, "use_rim_only", 0, NULL, ICON_NONE);

  uiItemS(layout);

  modifier_vgroup_ui(layout, ptr, &ob_ptr, "vertex_group", "invert_vertex_group", NULL);
  row = uiLayoutRow(layout, false);
  uiLayoutSetActive(row, has_vertex_group);
  uiItemR(row, ptr, "thickness_vertex_group", 0, IFACE_("Factor"), ICON_NONE);

  if (solidify_mode == MOD_SOLIDIFY_MODE_NONMANIFOLD) {
    row = uiLayoutRow(layout, false);
    uiLayoutSetActive(row, has_vertex_group);
    uiItemR(row, ptr, "use_flat_faces", 0, NULL, ICON_NONE);
  }

  modifier_panel_end(layout, ptr);
}

static void normals_panel_draw(const bContext *UNUSED(C), Panel *panel)
{
  uiLayout *col;
  uiLayout *layout = panel->layout;

  PointerRNA ob_ptr;
  PointerRNA *ptr = modifier_panel_get_property_pointers(panel, &ob_ptr);

  int solidify_mode = RNA_enum_get(ptr, "solidify_mode");

  uiLayoutSetPropSep(layout, true);

  col = uiLayoutColumn(layout, false);
  uiItemR(col, ptr, "use_flip_normals", 0, IFACE_("Flip"), ICON_NONE);
  if (solidify_mode == MOD_SOLIDIFY_MODE_EXTRUDE) {
    uiItemR(col, ptr, "use_quality_normals", 0, IFACE_("High Quality"), ICON_NONE);
  }
}

static void materials_panel_draw(const bContext *UNUSED(C), Panel *panel)
{
  uiLayout *col;
  uiLayout *layout = panel->layout;

  PointerRNA ob_ptr;
  PointerRNA *ptr = modifier_panel_get_property_pointers(panel, &ob_ptr);

  uiLayoutSetPropSep(layout, true);

  uiItemR(layout, ptr, "material_offset", 0, NULL, ICON_NONE);
  col = uiLayoutColumn(layout, true);
  uiLayoutSetActive(col, RNA_boolean_get(ptr, "use_rim"));
  uiItemR(col, ptr, "material_offset_rim", 0, IFACE_("Rim"), ICON_NONE);
}

static void edge_data_panel_draw(const bContext *UNUSED(C), Panel *panel)
{
  uiLayout *layout = panel->layout;

  PointerRNA ob_ptr;
  PointerRNA *ptr = modifier_panel_get_property_pointers(panel, &ob_ptr);

  int solidify_mode = RNA_enum_get(ptr, "solidify_mode");

  uiLayoutSetPropSep(layout, true);

  if (solidify_mode == MOD_SOLIDIFY_MODE_EXTRUDE) {
    uiLayout *col;
    col = uiLayoutColumn(layout, true);
    uiItemR(col, ptr, "edge_crease_inner", 0, IFACE_("Crease Inner"), ICON_NONE);
    uiItemR(col, ptr, "edge_crease_outer", 0, IFACE_("Outer"), ICON_NONE);
    uiItemR(col, ptr, "edge_crease_rim", 0, IFACE_("Rim"), ICON_NONE);
  }
  uiItemR(layout, ptr, "bevel_convex", UI_ITEM_R_SLIDER, NULL, ICON_NONE);
}

static void clamp_panel_draw(const bContext *UNUSED(C), Panel *panel)
{
  uiLayout *row, *col;
  uiLayout *layout = panel->layout;

  PointerRNA ob_ptr;
  PointerRNA *ptr = modifier_panel_get_property_pointers(panel, &ob_ptr);

  uiLayoutSetPropSep(layout, true);

  col = uiLayoutColumn(layout, false);
  uiItemR(col, ptr, "thickness_clamp", 0, NULL, ICON_NONE);
  row = uiLayoutRow(col, false);
  uiLayoutSetActive(row, RNA_float_get(ptr, "thickness_clamp") > 0.0f);
  uiItemR(row, ptr, "use_thickness_angle_clamp", 0, NULL, ICON_NONE);
}

static void vertex_group_panel_draw(const bContext *UNUSED(C), Panel *panel)
{
  uiLayout *col;
  uiLayout *layout = panel->layout;

  PointerRNA ob_ptr;
  PointerRNA *ptr = modifier_panel_get_property_pointers(panel, &ob_ptr);

  uiLayoutSetPropSep(layout, true);

  col = uiLayoutColumn(layout, false);
  uiItemPointerR(
      col, ptr, "shell_vertex_group", &ob_ptr, "vertex_groups", IFACE_("Shell"), ICON_NONE);
  uiItemPointerR(col, ptr, "rim_vertex_group", &ob_ptr, "vertex_groups", IFACE_("Rim"), ICON_NONE);
}

static void panelRegister(ARegionType *region_type)
{
  PanelType *panel_type = modifier_panel_register(region_type, eModifierType_Solidify, panel_draw);
  modifier_subpanel_register(
      region_type, "normals", "Normals", NULL, normals_panel_draw, panel_type);
  modifier_subpanel_register(
      region_type, "materials", "Materials", NULL, materials_panel_draw, panel_type);
  modifier_subpanel_register(
      region_type, "edge_data", "Edge Data", NULL, edge_data_panel_draw, panel_type);
  modifier_subpanel_register(
      region_type, "clamp", "Thickness Clamp", NULL, clamp_panel_draw, panel_type);
  modifier_subpanel_register(region_type,
                             "vertex_groups",
                             "Output Vertex Groups",
                             NULL,
                             vertex_group_panel_draw,
                             panel_type);
}

ModifierTypeInfo modifierType_Solidify = {
    /* name */ "Solidify",
    /* structName */ "SolidifyModifierData",
    /* structSize */ sizeof(SolidifyModifierData),
    /* srna */ &RNA_SolidifyModifier,
    /* type */ eModifierTypeType_Constructive,

    /* flags */ eModifierTypeFlag_AcceptsMesh | eModifierTypeFlag_AcceptsCVs |
        eModifierTypeFlag_SupportsMapping | eModifierTypeFlag_SupportsEditmode |
        eModifierTypeFlag_EnableInEditmode,
    /* icon */ ICON_MOD_SOLIDIFY,

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
    /* dependsOnNormals */ dependsOnNormals,
    /* foreachIDLink */ NULL,
    /* foreachTexLink */ NULL,
    /* freeRuntimeData */ NULL,
    /* panelRegister */ panelRegister,
    /* blendWrite */ NULL,
    /* blendRead */ NULL,
};
