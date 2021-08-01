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
#  include <openvdb/tools/LevelSetFilter.h>
#endif

#include "BKE_volume.h"

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "node_geometry_util.hh"

static bNodeSocketTemplate geo_node_level_set_filter_in[] = {
    {SOCK_GEOMETRY, N_("Level Set")},
    {SOCK_FLOAT, N_("Distance"), 0.1f, 0.0f, 0.0f, 0.0f, -FLT_MAX, FLT_MAX, PROP_DISTANCE},
    {SOCK_INT, N_("Width"), 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 128, PROP_DISTANCE},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_level_set_filter_out[] = {
    {SOCK_GEOMETRY, N_("Level Set")},
    {-1, ""},
};

static void geo_node_level_set_filter_layout(uiLayout *layout,
                                             bContext *UNUSED(C),
                                             PointerRNA *ptr)
{
  uiItemR(layout, ptr, "operation", 0, "", ICON_NONE);
}

static void geo_node_level_set_filter_init(bNodeTree *UNUSED(ntree), bNode *node)
{
  NodeGeometryLevelSetFilter *data = (NodeGeometryLevelSetFilter *)MEM_callocN(
      sizeof(NodeGeometryLevelSetFilter), __func__);
  data->operation = GEO_NODE_LEVEL_SET_FILTER_OFFSET;
  node->storage = data;
}

static void geo_node_level_set_filter_update(bNodeTree *UNUSED(ntree), bNode *node)
{
  const NodeGeometryLevelSetFilter &data = *(const NodeGeometryLevelSetFilter *)node->storage;
  const GeometryNodeFilterOperation operation = (GeometryNodeFilterOperation)data.operation;

  bNodeSocket *distance_socket = ((bNodeSocket *)node->inputs.first)->next;
  bNodeSocket *width_socket = distance_socket->next;

  nodeSetSocketAvailability(distance_socket, operation == GEO_NODE_LEVEL_SET_FILTER_OFFSET);
  nodeSetSocketAvailability(width_socket,
                            ELEM(operation,
                                 GEO_NODE_LEVEL_SET_FILTER_GAUSSIAN,
                                 GEO_NODE_LEVEL_SET_FILTER_MEDIAN,
                                 GEO_NODE_LEVEL_SET_FILTER_MEAN));
}

namespace blender::nodes {

#ifdef WITH_OPENVDB

struct FilterGridOp {
  openvdb::GridBase &grid_base;
  GeometryNodeFilterOperation operation;
  const GeoNodeExecParams &params;

  template<typename GridType> void operator()()
  {
    if constexpr (std::is_same_v<GridType, openvdb::FloatGrid>) {
      this->filter_operation<GridType>();
    }
    else {
      params.error_message_add(NodeWarningType::Error,
                               TIP_("Filter operations only support float grids"));
    }
  }

  template<typename GridType> void filter_operation()
  {
    GridType &grid = static_cast<GridType &>(grid_base);

    openvdb::tools::LevelSetFilter<GridType> filter(grid);
    switch (operation) {
      case GEO_NODE_LEVEL_SET_FILTER_GAUSSIAN:
        filter.gaussian(params.get_input<int>("Width"));
        break;
      case GEO_NODE_LEVEL_SET_FILTER_OFFSET:
        filter.offset(-params.get_input<float>("Distance"));
        break;
      case GEO_NODE_LEVEL_SET_FILTER_MEDIAN:
        filter.median(params.get_input<int>("Width"));
        break;
      case GEO_NODE_LEVEL_SET_FILTER_MEAN:
        filter.mean(params.get_input<int>("Width"));
        break;
      case GEO_NODE_LEVEL_SET_FILTER_MEAN_CURVATURE:
        filter.meanCurvature();
        break;
      case GEO_NODE_LEVEL_SET_FILTER_LAPLACIAN:
        filter.laplacian();
        break;
    }
  }
};

static void level_set_filter(Volume &volume,
                             const GeometryNodeFilterOperation operation,
                             const GeoNodeExecParams &params)
{
  VolumeGrid *volume_grid = BKE_volume_grid_get_for_write(&volume, 0);
  if (volume_grid == nullptr) {
    params.error_message_add(NodeWarningType::Error, TIP_("Volume is empty"));
    return;
  }

  openvdb::GridBase::Ptr grid = BKE_volume_grid_openvdb_for_write(&volume, volume_grid);

  FilterGridOp filter_grid_op{*grid, operation, params};
  BKE_volume_grid_type_operation(BKE_volume_grid_type(volume_grid), filter_grid_op);
}

#endif /* WITH_OPENVDB */

static void geo_node_level_set_filter_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Level Set");

#ifdef WITH_OPENVDB
  Volume *volume = geometry_set.get_volume_for_write();
  if (volume == nullptr) {
    params.set_output("Level Set", std::move(geometry_set));
    return;
  }

  const NodeGeometryLevelSetFilter &data =
      *(const NodeGeometryLevelSetFilter *)params.node().storage;
  const GeometryNodeFilterOperation operation = (GeometryNodeFilterOperation)data.operation;

  level_set_filter(*volume, operation, params);
#endif

  params.set_output("Level Set", std::move(geometry_set));
}

}  // namespace blender::nodes

void register_node_type_geo_level_set_filter()
{
  static bNodeType ntype;

  geo_node_type_base(
      &ntype, GEO_NODE_LEVEL_SET_FILTER, "Level Set Filter", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, geo_node_level_set_filter_in, geo_node_level_set_filter_out);
  ntype.geometry_node_execute = blender::nodes::geo_node_level_set_filter_exec;
  node_type_storage(&ntype,
                    "NodeGeometryLevelSetFilter",
                    node_free_standard_storage,
                    node_copy_standard_storage);
  node_type_init(&ntype, geo_node_level_set_filter_init);
  ntype.draw_buttons = geo_node_level_set_filter_layout;
  ntype.updatefunc = geo_node_level_set_filter_update;

  nodeRegisterType(&ntype);
}
