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
#  include <openvdb/tools/LevelSetPlatonic.h>
#endif

#include "BKE_lib_id.h"
#include "BKE_volume.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "node_geometry_util.hh"

static bNodeSocketTemplate geo_node_level_set_primitive_platonic_in[] = {
    {SOCK_FLOAT, N_("Size"), 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, FLT_MAX, PROP_DISTANCE},
    {SOCK_VECTOR, N_("Center"), 0.0f, 0.0f, 0.0f, 0.0f, -FLT_MAX, FLT_MAX, PROP_TRANSLATION},
    {SOCK_FLOAT, N_("Voxel Size"), 0.3f, 0.0f, 0.0f, 0.0f, 0.01f, FLT_MAX, PROP_DISTANCE},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_level_set_primitive_platonic_out[] = {
    {SOCK_GEOMETRY, N_("Level Set")},
    {-1, ""},
};

static void geo_node_level_set_primitive_platonic_layout(uiLayout *layout,
                                                         bContext *UNUSED(C),
                                                         PointerRNA *ptr)
{
  uiItemR(layout, ptr, "shape", 0, "", ICON_NONE);
}

static void geo_node_level_set_primitive_platonic_init(bNodeTree *UNUSED(ntree), bNode *node)
{
  NodeGeometryLevelSetPlatonic *data = (NodeGeometryLevelSetPlatonic *)MEM_callocN(
      sizeof(NodeGeometryLevelSetPlatonic), __func__);
  data->shape = GEO_NODE_PLATONIC_CUBE;
  node->storage = data;
}

namespace blender::nodes {

#ifdef WITH_OPENVDB

static Volume *level_set_primitive_platonic(const NodeGeometryPlatonicShape shape,
                                            GeoNodeExecParams &params)
{
  Volume *volume = (Volume *)BKE_id_new_nomain(ID_VO, nullptr);
  BKE_volume_init_grids(volume);

  const float3 center = params.get_input<float3>("Center");
  openvdb::FloatGrid::Ptr grid = openvdb::tools::createLevelSetPlatonic<openvdb::FloatGrid>(
      static_cast<int>(shape),
      params.extract_input<float>("Size"),
      openvdb::Vec3f(center.x, center.y, center.z),
      params.extract_input<float>("Voxel Size"));

  BKE_volume_grid_add_vdb(volume, "level_set", std::move(grid));

  return volume;
}

#endif /* WITH_OPENVDB */

static void geo_node_level_set_primitive_platonic_exec(GeoNodeExecParams params)
{
#ifdef WITH_OPENVDB
  const NodeGeometryLevelSetPlatonic &data =
      *(const NodeGeometryLevelSetPlatonic *)params.node().storage;
  const NodeGeometryPlatonicShape shape = (NodeGeometryPlatonicShape)data.shape;

  Volume *volume = level_set_primitive_platonic(shape, params);
  params.set_output("Level Set", GeometrySet::create_with_volume(volume));
#else
  params.set_output("Level Set", GeometrySet());
#endif
}

}  // namespace blender::nodes

void register_node_type_geo_level_set_primitive_platonic()
{
  static bNodeType ntype;

  geo_node_type_base(
      &ntype, GEO_NODE_LEVEL_SET_PRIMITIVE_PLATONIC, "Level Set Platonic", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(
      &ntype, geo_node_level_set_primitive_platonic_in, geo_node_level_set_primitive_platonic_out);
  ntype.geometry_node_execute = blender::nodes::geo_node_level_set_primitive_platonic_exec;
  node_type_storage(&ntype,
                    "NodeGeometryLevelSetPlatonic",
                    node_free_standard_storage,
                    node_copy_standard_storage);
  node_type_init(&ntype, geo_node_level_set_primitive_platonic_init);
  ntype.draw_buttons = geo_node_level_set_primitive_platonic_layout;

  nodeRegisterType(&ntype);
}
