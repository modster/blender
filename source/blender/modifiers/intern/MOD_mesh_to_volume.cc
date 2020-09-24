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
#include "BKE_object.h"
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
#include "BLI_span.hh"

#ifdef WITH_OPENVDB
#  include <openvdb/openvdb.h>
#  include <openvdb/tools/MeshToVolume.h>
#endif

#ifdef WITH_OPENVDB
namespace blender {
class OpenVDBMeshAdapter {
 private:
  Span<MVert> vertices_;
  Span<MLoop> loops_;
  Span<MLoopTri> looptris_;
  float4x4 transform_;

 public:
  OpenVDBMeshAdapter(Mesh &mesh, float4x4 transform)
      : vertices_(mesh.mvert, mesh.totvert),
        loops_(mesh.mloop, mesh.totloop),
        transform_(transform)
  {
    const MLoopTri *looptries = BKE_mesh_runtime_looptri_ensure(&mesh);
    const int looptries_len = BKE_mesh_runtime_looptri_len(&mesh);
    looptris_ = Span(looptries, looptries_len);
  }

  size_t polygonCount() const
  {
    return static_cast<size_t>(looptris_.size());
  }

  size_t pointCount() const
  {
    return static_cast<size_t>(vertices_.size());
  }

  size_t vertexCount(size_t UNUSED(polygon_index)) const
  {
    /* All polygons are triangles. */
    return 3;
  }

  void getIndexSpacePoint(size_t polygon_index, size_t vertex_index, openvdb::Vec3d &pos) const
  {
    const MLoopTri &looptri = looptris_[polygon_index];
    const MVert &vertex = vertices_[loops_[looptri.tri[vertex_index]].v];
    const float3 transformed_co = transform_ * float3(vertex.co);
    pos = &transformed_co.x;
  }
};
}  // namespace blender
#endif

static void initData(ModifierData *md)
{
  MeshToVolumeModifierData *mvmd = reinterpret_cast<MeshToVolumeModifierData *>(md);
  mvmd->object = NULL;
  mvmd->mode = MESH_TO_VOLUME_MODE_VOLUME;
  mvmd->resolution_mode = MESH_TO_VOLUME_RESOLUTION_MODE_VOXEL_AMOUNT;
  mvmd->voxel_size = 0.1f;
  mvmd->voxel_amount = 32;
  mvmd->fill_volume = true;
  mvmd->interior_bandwidth = 1.0f;
  mvmd->exterior_bandwidth = 1.0f;
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
  MeshToVolumeModifierData *mvmd = static_cast<MeshToVolumeModifierData *>(ptr->data);

  uiLayoutSetPropSep(layout, true);
  uiLayoutSetPropDecorate(layout, false);

  uiItemR(layout, ptr, "object", 0, NULL, ICON_NONE);

  uiItemR(layout, ptr, "mode", 0, NULL, ICON_NONE);
  if (mvmd->mode == MESH_TO_VOLUME_MODE_VOLUME) {
    uiItemR(layout, ptr, "fill_volume", 0, NULL, ICON_NONE);
    uiItemR(layout, ptr, "exterior_bandwidth", 0, NULL, ICON_NONE);
    if (!mvmd->fill_volume) {
      uiItemR(layout, ptr, "interior_bandwidth", 0, NULL, ICON_NONE);
    }
  }
  else if (mvmd->mode == MESH_TO_VOLUME_MODE_SURFACE) {
    uiItemR(layout, ptr, "exterior_bandwidth", 0, "Bandwidth", ICON_NONE);
  }

  uiItemR(layout, ptr, "resolution_mode", 0, NULL, ICON_NONE);
  if (mvmd->resolution_mode == MESH_TO_VOLUME_RESOLUTION_MODE_VOXEL_AMOUNT) {
    uiItemR(layout, ptr, "voxel_amount", 0, NULL, ICON_NONE);
  }
  else {
    uiItemR(layout, ptr, "voxel_size", 0, NULL, ICON_NONE);
  }

  modifier_panel_end(layout, ptr);
}

static void panelRegister(ARegionType *region_type)
{
  modifier_panel_register(region_type, eModifierType_MeshToVolume, panel_draw);
}

static float compute_voxel_size(const MeshToVolumeModifierData *mvmd,
                                const blender::float4x4 &transform)
{
  using namespace blender;

  if (mvmd->resolution_mode == MESH_TO_VOLUME_RESOLUTION_MODE_VOXEL_SIZE) {
    return MAX2(0.0001, mvmd->voxel_size);
  }
  BoundBox *bb = BKE_object_boundbox_get(mvmd->object);

  float3 dimensions = float3(bb->vec[6]) - float3(bb->vec[0]);
  const float3 transformed_dimensions = transform.ref_3x3() * dimensions;
  const float max_dimension = std::max(
      {transformed_dimensions.x, transformed_dimensions.y, transformed_dimensions.z});
  const float approximate_volume_side_length = max_dimension + mvmd->exterior_bandwidth * 2.0f;
  const float voxel_size = approximate_volume_side_length / MAX2(1, mvmd->voxel_amount);
  return voxel_size;
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

  Object *object_to_convert = mvmd->object;
  const float4x4 mesh_to_own_object_space_transform = float4x4(ctx->object->imat) *
                                                      float4x4(object_to_convert->obmat);
  const float voxel_size = compute_voxel_size(mvmd, mesh_to_own_object_space_transform);

  float4x4 mesh_to_index_space_transform;
  scale_m4_fl(mesh_to_index_space_transform.values, 1.0f / voxel_size);
  mul_m4_m4_post(mesh_to_index_space_transform.values, mesh_to_own_object_space_transform.values);

  Mesh *mesh = static_cast<Mesh *>(object_to_convert->data);
  OpenVDBMeshAdapter mesh_adapter{*mesh, mesh_to_index_space_transform};

  openvdb::FloatGrid::Ptr new_grid;

  const float exterior_bandwidth = MAX2(0.001f, mvmd->exterior_bandwidth / voxel_size);
  const float interior_bandwidth = MAX2(0.001f, mvmd->interior_bandwidth / voxel_size);
  if (mvmd->mode == MESH_TO_VOLUME_MODE_VOLUME) {
    if (mvmd->fill_volume) {
      new_grid = openvdb::tools::meshToVolume<openvdb::FloatGrid>(
          mesh_adapter, {}, exterior_bandwidth, FLT_MAX);
    }
    else {
      new_grid = openvdb::tools::meshToVolume<openvdb::FloatGrid>(
          mesh_adapter, {}, exterior_bandwidth, interior_bandwidth);
    }
  }
  else {
    new_grid = openvdb::tools::meshToVolume<openvdb::FloatGrid>(
        mesh_adapter,
        {},
        exterior_bandwidth,
        exterior_bandwidth,
        openvdb::tools::UNSIGNED_DISTANCE_FIELD);
  }

  Volume *volume = BKE_volume_new_for_eval(input_volume);
  VolumeGrid *c_density_grid = BKE_volume_grid_add(volume, "density", VOLUME_GRID_FLOAT);
  openvdb::FloatGrid::Ptr density_grid = std::static_pointer_cast<openvdb::FloatGrid>(
      BKE_volume_grid_openvdb_for_write(volume, c_density_grid, false));
  density_grid->merge(*new_grid);
  density_grid->transform().postScale(voxel_size);

  openvdb::tools::foreach (
      density_grid->beginValueOn(),
      [](const openvdb::FloatGrid::ValueOnIter &iter) { iter.setValue(1.0f); });

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
