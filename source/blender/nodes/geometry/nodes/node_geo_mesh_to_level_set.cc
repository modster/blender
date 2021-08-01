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

#ifdef WITH_OPENVDB
#  include <openvdb/tools/GridTransformer.h>
#  include <openvdb/tools/VolumeToMesh.h>
#endif

#include "BKE_lib_id.h"
#include "BKE_mesh.h"
#include "BKE_mesh_runtime.h"
#include "BKE_volume.h"

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "node_geometry_util.hh"

using blender::bke::GeometryInstanceGroup;

static bNodeSocketTemplate geo_node_mesh_to_level_set_in[] = {
    {SOCK_GEOMETRY, N_("Mesh")},
    {SOCK_FLOAT, N_("Voxel Size"), 0.3f, 0.0f, 0.0f, 0.0f, 0.01f, FLT_MAX, PROP_DISTANCE},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_mesh_to_level_set_out[] = {
    {SOCK_GEOMETRY, N_("Level Set")},
    {-1, ""},
};

namespace blender::nodes {

#ifdef WITH_OPENVDB

static openvdb::FloatGrid::Ptr meshes_to_level_set_grid(
    const Span<GeometryInstanceGroup> set_groups, const float voxel_size)
{
  const float voxel_size_inv = 1.0f / voxel_size;

  /* Count the vertex and triangle size of all input meshes. */
  int vert_size = 0;
  int tri_size = 0;
  for (const GeometryInstanceGroup &set_group : set_groups) {
    const GeometrySet &set = set_group.geometry_set;
    const Mesh &mesh = *set.get_mesh_for_read();

    vert_size += mesh.totvert * set_group.transforms.size();
    tri_size += BKE_mesh_runtime_looptri_len(&mesh) * set_group.transforms.size();
  }

  int vert_index = 0;
  int tri_index = 0;
  std::vector<openvdb::Vec3s> positions(vert_size);
  std::vector<openvdb::Vec3I> triangles(tri_size);
  for (const GeometryInstanceGroup &set_group : set_groups) {
    const GeometrySet &set = set_group.geometry_set;
    const Mesh &mesh = *set.get_mesh_for_read();
    const Span<MLoopTri> looptris{BKE_mesh_runtime_looptri_ensure(&mesh),
                                  BKE_mesh_runtime_looptri_len(&mesh)};
    const Span<MLoop> mloop{mesh.mloop, mesh.totloop};
    for (const float4x4 &transform : set_group.transforms) {
      const int vert_offset = vert_index;
      for (const int i : IndexRange(mesh.totvert)) {
        const float3 co = transform * float3(mesh.mvert[i].co);
        /* Better align generated grid with source points. */
        const float3 index_co = co * voxel_size_inv - float3(0.5f);
        positions[vert_index++] = openvdb::Vec3s(index_co.x, index_co.y, index_co.z);
      }
      for (const int i : IndexRange(looptris.size())) {
        const MLoopTri &loop_tri = looptris[i];
        triangles[tri_index++] = openvdb::Vec3I(vert_offset + mloop[loop_tri.tri[0]].v,
                                                vert_offset + mloop[loop_tri.tri[1]].v,
                                                vert_offset + mloop[loop_tri.tri[2]].v);
      }
    }
  }

  // openvdb::math::Transform::Ptr transform = openvdb::math::Transform::createLinearTransform(
  //     voxel_size);
  openvdb::FloatGrid::Ptr grid = openvdb::tools::meshToLevelSet<openvdb::FloatGrid>(
      {}, positions, triangles, 1.0f);
  grid->transform().postScale(voxel_size);

  return grid;
}

static Volume *meshes_to_level_set_volume(const Span<GeometryInstanceGroup> set_groups,
                                          const float voxel_size)
{
  Volume *volume = (Volume *)BKE_id_new_nomain(ID_VO, nullptr);
  BKE_volume_init_grids(volume);

  openvdb::FloatGrid::Ptr new_grid = meshes_to_level_set_grid(set_groups, voxel_size);
  // new_grid->transform().postScale(voxel_size);

  BKE_volume_grid_add_vdb(volume, "level_set", new_grid);

  return volume;
}

#endif /* WITH_OPENVDB */

static void geo_node_mesh_to_level_set_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Mesh");

  Vector<GeometryInstanceGroup> set_groups;
  bke::geometry_set_gather_instances(geometry_set, set_groups);
  if (set_groups.is_empty()) {
    params.set_output("Level Set", GeometrySet());
    return;
  }

  /* Remove any set inputs that don't contain a mesh, to avoid checking later on. */
  for (int i = set_groups.size() - 1; i >= 0; i--) {
    const GeometrySet &set = set_groups[i].geometry_set;
    if (!set.has_mesh()) {
      set_groups.remove_and_reorder(i);
    }
  }

  GeometrySet geometry_set_out;
#ifdef WITH_OPENVDB
  const float voxel_size = params.get_input<float>("Voxel Size");
  Volume *volume = meshes_to_level_set_volume(set_groups, voxel_size);
#endif

  params.set_output("Level Set", GeometrySet::create_with_volume(volume));
}

}  // namespace blender::nodes

void register_node_type_geo_mesh_to_level_set()
{
  static bNodeType ntype;

  geo_node_type_base(
      &ntype, GEO_NODE_MESH_TO_LEVEL_SET, "Mesh to Level Set", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(
      &ntype, geo_node_mesh_to_level_set_in, geo_node_mesh_to_level_set_out);
  ntype.geometry_node_execute = blender::nodes::geo_node_mesh_to_level_set_exec;
  nodeRegisterType(&ntype);
}
