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

#include "BKE_derived_curve.hh"

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

// /* Set the location for the first point. */
// r_samples[0].x = profile->table[0].x;
// r_samples[0].y = profile->table[0].y;

// /* Travel along the path, recording the locations of segments as we pass them. */
// float segment_left = segment_length;
// for (int i = 1; i < n_segments; i++) {
// /* Travel over all of the points that fit inside this segment. */
// while (distance_to_next_table_point < segment_left) {
//     length_travelled += distance_to_next_table_point;
//     segment_left -= distance_to_next_table_point;
//     i_table++;
//     distance_to_next_table_point = curveprofile_distance_to_next_table_point(profile, i_table);
//     distance_to_previous_table_point = 0.0f;
// }
// /* We're at the last table point that fits inside the current segment, use interpolation. */
// float factor = (distance_to_previous_table_point + segment_left) /
//                 (distance_to_previous_table_point + distance_to_next_table_point);
// r_samples[i].x = interpf(profile->table[i_table + 1].x, profile->table[i_table].x, factor);
// r_samples[i].y = interpf(profile->table[i_table + 1].y, profile->table[i_table].y, factor);
// BLI_assert(factor <= 1.0f && factor >= 0.0f);
// #ifdef DEBUG_CURVEPROFILE_EVALUATE
// printf("segment_left: %.3f\n", segment_left);
// printf("i_table: %d\n", i_table);
// printf("distance_to_previous_table_point: %.3f\n", distance_to_previous_table_point);
// printf("distance_to_next_table_point: %.3f\n", distance_to_next_table_point);
// printf("Interpolating with factor %.3f from (%.3f, %.3f) to (%.3f, %.3f)\n\n",
//         factor,
//         profile->table[i_table].x,
//         profile->table[i_table].y,
//         profile->table[i_table + 1].x,
//         profile->table[i_table + 1].y);
// #endif

// /* We sampled in between this table point and the next, so the next travel step is smaller. */
// distance_to_next_table_point -= segment_left;
// distance_to_previous_table_point += segment_left;
// length_travelled += segment_left;
// segment_left = segment_length;
// }

static void geo_node_curve_sample_points_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Curve");

  const bNode &node = params.node();
  const NodeGeometryCurveSamplePoints &node_storage = *(const NodeGeometryCurveSamplePoints *)
                                                           node.storage;
  const GeometryNodeCurveSamplePointsMode mode = (GeometryNodeCurveSamplePointsMode)
                                                     node_storage.mode;

  params.error_message_add(NodeWarningType::Info, "The node doesn't do anything yet");

  if (!geometry_set.has_curve()) {
    params.set_output("Points", geometry_set);
  }

  const DCurve &curve = *geometry_set.get_curve_for_read();

  switch (mode) {
    case GEO_NODE_CURVE_SAMPLE_POINTS_COUNT: {
      break;
    }
    case GEO_NODE_CURVE_SAMPLE_POINTS_LENGTH: {
      break;
    }
    default:
      BLI_assert_unreachable();
      break;
  }

  params.set_output("Geometry", geometry_set);
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
