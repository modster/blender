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
#  include <openvdb/openvdb.h>
#  include <openvdb/tools/Composite.h>
#  include <openvdb/tools/GridTransformer.h>
#endif

#include "BKE_volume.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "node_geometry_util.hh"

static bNodeSocketTemplate geo_node_level_set_boolean_in[] = {
    {SOCK_GEOMETRY, N_("Level Set 1")},
    {SOCK_GEOMETRY, N_("Level Set 2")},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_level_set_boolean_out[] = {
    {SOCK_GEOMETRY, N_("Level Set")},
    {-1, ""},
};

static void geo_node_level_set_boolean_layout(uiLayout *layout,
                                              bContext *UNUSED(C),
                                              PointerRNA *ptr)
{
  uiItemR(layout, ptr, "operation", 0, "", ICON_NONE);
}

static void geo_node_level_set_boolean_init(bNodeTree *UNUSED(ntree), bNode *node)
{
  NodeGeometryLevelSetBoolean *data = (NodeGeometryLevelSetBoolean *)MEM_callocN(
      sizeof(NodeGeometryLevelSetBoolean), __func__);
  data->operation = GEO_NODE_BOOLEAN_UNION;
  node->storage = data;
}

namespace blender::nodes {

#ifdef WITH_OPENVDB

static void level_set_boolean(Volume &volume_a,
                              const Volume &volume_b,
                              const GeometryNodeBooleanOperation operation,
                              const GeoNodeExecParams &params)
{
  VolumeGrid *volume_grid_a = BKE_volume_grid_get_for_write(&volume_a, 0);
  const VolumeGrid *volume_grid_b = BKE_volume_grid_get_for_read(&volume_b, 0);
  if (ELEM(nullptr, volume_grid_a, volume_grid_b)) {
    if (volume_grid_a == nullptr) {
      params.error_message_add(NodeWarningType::Info, TIP_("Volume 1 is empty"));
    }
    if (volume_grid_b == nullptr) {
      params.error_message_add(NodeWarningType::Info, TIP_("Volume 2 is empty"));
    }
    return;
  }
  openvdb::GridBase::Ptr grid_base_a = BKE_volume_grid_openvdb_for_write(&volume_a, volume_grid_a);
  openvdb::GridBase::ConstPtr grid_base_b = BKE_volume_grid_openvdb_for_read(&volume_b,
                                                                             volume_grid_b);
  if (grid_base_a->getGridClass() != openvdb::GridClass::GRID_LEVEL_SET ||
      grid_base_b->getGridClass() != openvdb::GridClass::GRID_LEVEL_SET) {
    if (grid_base_a->getGridClass() != openvdb::GridClass::GRID_LEVEL_SET) {
      params.error_message_add(NodeWarningType::Error, TIP_("Volume 1 is not a level set"));
    }
    if (grid_base_b->getGridClass() != openvdb::GridClass::GRID_LEVEL_SET) {
      params.error_message_add(NodeWarningType::Error, TIP_("Volume 2 is not a level set"));
    }
    return;
  }

  const VolumeGridType grid_type_a = BKE_volume_grid_type(volume_grid_a);
  const VolumeGridType grid_type_b = BKE_volume_grid_type(volume_grid_b);
  if (grid_type_a != grid_type_b) {
    params.error_message_add(NodeWarningType::Error, TIP_("Volume grid types do not match"));
    return;
  }

  bke::volume::to_static_type(grid_type_a, [&](auto dummy) {
    using GridType = decltype(dummy);
    if constexpr (std::is_scalar<typename GridType::ValueType>::value) {
      GridType &grid_a = static_cast<GridType &>(*grid_base_a);
      const GridType &grid_b = static_cast<const GridType &>(*grid_base_b);

      openvdb::GridBase::Ptr grid_b_resampled_base = GridType::create();
      GridType &grid_b_resampled = static_cast<GridType &>(*grid_b_resampled_base);
      grid_b_resampled.setTransform(grid_base_a->constTransform().copy());

      openvdb::tools::resampleToMatch<openvdb::tools::BoxSampler>(grid_b, grid_b_resampled);
      openvdb::tools::pruneLevelSet(grid_b_resampled.tree());

      switch (operation) {
        case GEO_NODE_BOOLEAN_INTERSECT:
          openvdb::tools::csgIntersection(grid_a, grid_b_resampled);
          break;
        case GEO_NODE_BOOLEAN_UNION:
          openvdb::tools::csgUnion(grid_a, grid_b_resampled);
          break;
        case GEO_NODE_BOOLEAN_DIFFERENCE:
          openvdb::tools::csgDifference(grid_a, grid_b_resampled);
          break;
      }
    }
  });
}

#endif /* WITH_OPENVDB */

static void geo_node_level_set_boolean_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set_a = params.extract_input<GeometrySet>("Level Set 1");
#ifdef WITH_OPENVDB
  GeometrySet geometry_set_b = params.extract_input<GeometrySet>("Level Set 2");

  const NodeGeometryLevelSetBoolean &storage =
      *(const NodeGeometryLevelSetBoolean *)params.node().storage;
  const GeometryNodeBooleanOperation operation = (GeometryNodeBooleanOperation)storage.operation;

  Volume *volume_a = geometry_set_a.get_volume_for_write();
  const Volume *volume_b = geometry_set_b.get_volume_for_read();
  if (volume_a == nullptr || volume_b == nullptr) {
    params.set_output("Level Set", std::move(geometry_set_a));
    return;
  }

  level_set_boolean(*volume_a, *volume_b, operation, params);
#else
  params.error_message_add(NodeWarningType::Error,
                           TIP_("Disabled, Blender was compiled without OpenVDB"));
#endif

  params.set_output("Level Set", std::move(geometry_set_a));
}

}  // namespace blender::nodes

void register_node_type_geo_level_set_boolean()
{
  static bNodeType ntype;

  geo_node_type_base(
      &ntype, GEO_NODE_LEVEL_SET_BOOLEAN, "Level Set Boolean", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(
      &ntype, geo_node_level_set_boolean_in, geo_node_level_set_boolean_out);
  ntype.geometry_node_execute = blender::nodes::geo_node_level_set_boolean_exec;
  node_type_storage(&ntype,
                    "NodeGeometryLevelSetBoolean",
                    node_free_standard_storage,
                    node_copy_standard_storage);
  node_type_init(&ntype, geo_node_level_set_boolean_init);
  ntype.draw_buttons = geo_node_level_set_boolean_layout;

  nodeRegisterType(&ntype);
}
