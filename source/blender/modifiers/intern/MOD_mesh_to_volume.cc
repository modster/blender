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
 */

/** \file
 * \ingroup modifiers
 */

#include <vector>

#include "BKE_lib_query.h"
#include "BKE_mesh_runtime.h"
#include "BKE_modifier.h"
#include "BKE_volume.h"

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"
#include "DNA_object_types.h"
#include "DNA_screen_types.h"
#include "DNA_volume_types.h"

#include "DEG_depsgraph_build.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "BLO_read_write.h"

#include "MEM_guardedalloc.h"

#include "MOD_modifiertypes.h"
#include "MOD_ui_common.h"

#include "BLI_float4x4.hh"
#include "BLI_index_range.hh"

#ifdef WITH_OPENVDB
#  include <openvdb/openvdb.h>
#  include <openvdb/tools/MeshToVolume.h>
#endif

static void initData(ModifierData *md)
{
  MeshToVolumeModifierData *mvmd = reinterpret_cast<MeshToVolumeModifierData *>(md);
  mvmd->object = NULL;
  mvmd->voxel_size = 0.1f;
}

static void updateDepsgraph(ModifierData *md, const ModifierUpdateDepsgraphContext *ctx)
{
  MeshToVolumeModifierData *mvmd = reinterpret_cast<MeshToVolumeModifierData *>(md);
  DEG_add_modifier_to_transform_relation(ctx->node, "own transforms");
  if (mvmd->object) {
    DEG_add_object_relation(
        ctx->node, mvmd->object, DEG_OB_COMP_GEOMETRY, "Object that is converted to a volume");
    DEG_add_object_relation(
        ctx->node, mvmd->object, DEG_OB_COMP_TRANSFORM, "Object that is converted to a volume");
  }
}

static void foreachObjectLink(ModifierData *md, Object *ob, ObjectWalkFunc walk, void *userData)
{
  MeshToVolumeModifierData *mvmd = reinterpret_cast<MeshToVolumeModifierData *>(md);

  walk(userData, ob, &mvmd->object, IDWALK_CB_NOP);
}

static void panel_draw(const bContext *UNUSED(C), Panel *panel)
{
  uiLayout *layout = panel->layout;

  PointerRNA ob_ptr;
  PointerRNA *ptr = modifier_panel_get_property_pointers(panel, &ob_ptr);

  uiLayoutSetPropSep(layout, true);
  uiLayoutSetPropDecorate(layout, false);

  uiItemR(layout, ptr, "object", 0, NULL, ICON_NONE);
  uiItemR(layout, ptr, "voxel_size", 0, NULL, ICON_NONE);

  modifier_panel_end(layout, ptr);
}

static void panelRegister(ARegionType *region_type)
{
  modifier_panel_register(region_type, eModifierType_MeshToVolume, panel_draw);
}

static Volume *modifyVolume(ModifierData *md, const ModifierEvalContext *ctx, Volume *input_volume)
{
#ifdef WITH_OPENVDB
  using namespace blender;

  MeshToVolumeModifierData *mvmd = reinterpret_cast<MeshToVolumeModifierData *>(md);

  if (mvmd->object == NULL) {
    return input_volume;
  }
  if (mvmd->object->type != OB_MESH) {
    return input_volume;
  }

  Mesh *mesh = static_cast<Mesh *>(mvmd->object->data);
  const MLoopTri *looptris = BKE_mesh_runtime_looptri_ensure(mesh);
  const int looptris_len = BKE_mesh_runtime_looptri_len(mesh);

  float4x4 obmat = mvmd->object->obmat;
  mul_m4_m4_pre(obmat.values, ctx->object->imat);
  mul_m4_fl(obmat.values, 1.0f / mvmd->voxel_size);
  obmat.values[3][3] = 1.0f;

  std::vector<openvdb::Vec3s> vertices(mesh->totvert);
  for (const int i : IndexRange(mesh->totvert)) {
    float3 position = obmat * float3(mesh->mvert[i].co);
    vertices[i] = &position.x;
  }

  std::vector<openvdb::Vec3I> triangle_indices(looptris_len);
  for (const int i : IndexRange(looptris_len)) {
    const MLoopTri &tri = looptris[i];
    triangle_indices[i] = {
        mesh->mloop[tri.tri[0]].v,
        mesh->mloop[tri.tri[1]].v,
        mesh->mloop[tri.tri[2]].v,
    };
  }

  const openvdb::math::Transform xform;
  openvdb::FloatGrid::Ptr new_grid = openvdb::tools::meshToLevelSet<openvdb::FloatGrid>(
      xform, vertices, triangle_indices);

  Volume *volume = BKE_volume_new_for_eval(input_volume);
  VolumeGrid *c_density_grid = BKE_volume_grid_add(volume, "density", VOLUME_GRID_FLOAT);
  openvdb::FloatGrid::Ptr density_grid = std::static_pointer_cast<openvdb::FloatGrid>(
      BKE_volume_grid_openvdb_for_write(volume, c_density_grid, false));
  density_grid->merge(*new_grid);
  density_grid->transform().postScale(mvmd->voxel_size);

  return volume;

#else
  UNUSED_VARS(md, ctx);
  return input_volume;
#endif
}

ModifierTypeInfo modifierType_MeshToVolume = {
    /* name */ "Mesh to Volume",
    /* structName */ "MeshToVolumeModifierData",
    /* structSize */ sizeof(MeshToVolumeModifierData),
    /* type */ eModifierTypeType_Constructive,
    /* flags */ static_cast<ModifierTypeFlag>(0),
    /* copyData */ BKE_modifier_copydata_generic,

    /* deformVerts */ NULL,
    /* deformMatrices */ NULL,
    /* deformVertsEM */ NULL,
    /* deformMatricesEM */ NULL,
    /* modifyMesh */ NULL,
    /* modifyHair */ NULL,
    /* modifyPointCloud */ NULL,
    /* modifyVolume */ modifyVolume,

    /* initData */ initData,
    /* requiredDataMask */ NULL,
    /* freeData */ NULL,
    /* isDisabled */ NULL,
    /* updateDepsgraph */ updateDepsgraph,
    /* dependsOnTime */ NULL,
    /* dependsOnNormals */ NULL,
    /* foreachObjectLink */ foreachObjectLink,
    /* foreachIDLink */ NULL,
    /* foreachTexLink */ NULL,
    /* freeRuntimeData */ NULL,
    /* panelRegister */ panelRegister,
    /* blendWrite */ NULL,
    /* blendRead */ NULL,
};
