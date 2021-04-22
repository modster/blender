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

#include "BKE_pointcloud.h"
#include "BKE_spline.hh"

#include "UI_interface.h"
#include "UI_resources.h"

#include "node_geometry_util.hh"

static bNodeSocketTemplate geo_node_curve_sample_points_in[] = {
    {SOCK_GEOMETRY, N_("Curve")},
    {SOCK_INT, N_("Count"), 10.0f, 0.0f, 0.0f, 0.0f, 0.0f, FLT_MAX},
    {SOCK_FLOAT, N_("Length"), 0.1f, 0.0f, 0.0f, 0.0f, 0.0f, FLT_MAX, PROP_DISTANCE},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_curve_sample_points_out[] = {
    {SOCK_GEOMETRY, N_("Points")},
    {-1, ""},
};

static void geo_node_curve_sample_points_layout(uiLayout *layout,
                                                bContext *UNUSED(C),
                                                PointerRNA *ptr)
{
  uiItemR(layout, ptr, "mode", UI_ITEM_R_EXPAND, nullptr, ICON_NONE);
}

static void geo_node_curve_sample_points_init(bNodeTree *UNUSED(tree), bNode *node)
{
  NodeGeometryCurveSamplePoints *data = (NodeGeometryCurveSamplePoints *)MEM_callocN(
      sizeof(NodeGeometryCurveSamplePoints), __func__);

  data->mode = GEO_NODE_CURVE_SAMPLE_POINTS_LENGTH;
  node->storage = data;
}

static void geo_node_curve_sample_points_update(bNodeTree *UNUSED(ntree), bNode *node)
{
  NodeGeometryCurveSamplePoints &node_storage = *(NodeGeometryCurveSamplePoints *)node->storage;
  const GeometryNodeCurveSamplePointsMode mode = (GeometryNodeCurveSamplePointsMode)
                                                     node_storage.mode;

  bNodeSocket *count_socket = ((bNodeSocket *)node->inputs.first)->next;
  bNodeSocket *length_socket = count_socket->next;

  nodeSetSocketAvailability(count_socket, mode == GEO_NODE_CURVE_SAMPLE_POINTS_COUNT);
  nodeSetSocketAvailability(length_socket, mode == GEO_NODE_CURVE_SAMPLE_POINTS_LENGTH);
}

namespace blender::nodes {

static void sample_points_from_spline(const Spline &spline,
                                      const int offset,
                                      const float sample_length,
                                      PointCloudComponent &point_component)
{
}

static Array<int> get_result_point_offsets(const DCurve &curve,
                                           const GeometryNodeCurveSamplePointsMode mode,
                                           const int count,
                                           const float length)
{
  Array<int> offsets(curve.splines.size() + 1);

  if (mode == GEO_NODE_CURVE_SAMPLE_POINTS_COUNT) {
    for (const int i : curve.splines.index_range()) {
      offsets[i] = count * i;
    }
    offsets.last() = curve.splines.size() * count;
  }
  else {
    int offset = 0;
    for (const int i : curve.splines.index_range()) {
      offsets[i] = offset;
      offset += curve.splines[i]->length() / length;
    }
    offsets.last() = offset;
  }

  return offsets;
}

static void geo_node_curve_sample_points_exec(GeoNodeExecParams params)
{
  const GeometrySet input_geometry_set = params.extract_input<GeometrySet>("Curve");

  const bNode &node = params.node();
  const NodeGeometryCurveSamplePoints &node_storage = *(const NodeGeometryCurveSamplePoints *)
                                                           node.storage;
  const GeometryNodeCurveSamplePointsMode mode = (GeometryNodeCurveSamplePointsMode)
                                                     node_storage.mode;

  if (!input_geometry_set.has_curve()) {
    params.set_output("Points", GeometrySet());
  }

  const DCurve &curve = *input_geometry_set.get_curve_for_read();

  const int count = (mode == GEO_NODE_CURVE_SAMPLE_POINTS_COUNT) ?
                        params.extract_input<int>("Count") :
                        0;
  const float length = (mode == GEO_NODE_CURVE_SAMPLE_POINTS_LENGTH) ?
                           params.extract_input<float>("Length") :
                           0.0f;

  Array<int> offsets = get_result_point_offsets(curve, mode, count, length);

  PointCloud *pointcloud = BKE_pointcloud_new_nomain(offsets.last());
  GeometrySet result_geometry_set = GeometrySet::create_with_pointcloud(pointcloud);
  PointCloudComponent &point_component =
      result_geometry_set.get_component_for_write<PointCloudComponent>();

  if (mode == GEO_NODE_CURVE_SAMPLE_POINTS_COUNT) {
    const int count = params.extract_input<int>("Count");
    for (const int i : curve.splines.index_range()) {
      Spline &spline = *curve.splines[i];
      sample_points_from_spline(spline, offsets[i], spline.length() / count, point_component);
    }
  }
  else {
  }

  params.set_output("Geometry", result_geometry_set);
}

}  // namespace blender::nodes

void register_node_type_geo_curve_sample_points()
{
  static bNodeType ntype;

  geo_node_type_base(
      &ntype, GEO_NODE_CURVE_SAMPLE_POINTS, "Sample Curve Points", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(
      &ntype, geo_node_curve_sample_points_in, geo_node_curve_sample_points_out);
  node_type_init(&ntype, geo_node_curve_sample_points_init);
  node_type_update(&ntype, geo_node_curve_sample_points_update);
  node_type_storage(
      &ntype, "NodeGeometryCurveTrim", node_free_standard_storage, node_copy_standard_storage);

  ntype.geometry_node_execute = blender::nodes::geo_node_curve_sample_points_exec;
  ntype.draw_buttons = geo_node_curve_sample_points_layout;
  nodeRegisterType(&ntype);
}
