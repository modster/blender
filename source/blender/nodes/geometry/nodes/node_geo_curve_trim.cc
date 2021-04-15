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

static bNodeSocketTemplate geo_node_curve_trim_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_FLOAT, N_("Start"), 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, PROP_FACTOR},
    {SOCK_FLOAT, N_("Start"), 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, FLT_MAX, PROP_DISTANCE},
    {SOCK_FLOAT, N_("End"), 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, PROP_FACTOR},
    {SOCK_FLOAT, N_("End"), 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, FLT_MAX, PROP_DISTANCE},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_curve_trim_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

static void geo_node_curve_trim_layout(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
{
  uiItemR(layout, ptr, "mode", UI_ITEM_R_EXPAND, nullptr, ICON_NONE);
}

static void geo_node_curve_trim_init(bNodeTree *UNUSED(tree), bNode *node)
{
  NodeGeometryCurveTrim *data = (NodeGeometryCurveTrim *)MEM_callocN(sizeof(NodeGeometryCurveTrim),
                                                                     __func__);

  data->mode = GEO_NODE_CURVE_TRIM_FACTOR;
  node->storage = data;
}

static void geo_node_curve_trim_update(bNodeTree *UNUSED(ntree), bNode *node)
{
  NodeGeometryCurveTrim &node_storage = *(NodeGeometryCurveTrim *)node->storage;
  const GeometryNodeCurveTrimMode mode = (GeometryNodeCurveTrimMode)node_storage.mode;

  bNodeSocket *factor_start_socket = ((bNodeSocket *)node->inputs.first)->next;
  bNodeSocket *length_start_socket = factor_start_socket->next;
  bNodeSocket *factor_end_socket = length_start_socket->next;
  bNodeSocket *length_end_socket = factor_end_socket->next;

  nodeSetSocketAvailability(factor_start_socket, mode == GEO_NODE_CURVE_TRIM_FACTOR);
  nodeSetSocketAvailability(length_start_socket, mode == GEO_NODE_CURVE_TRIM_LENGTH);
  nodeSetSocketAvailability(factor_end_socket, mode == GEO_NODE_CURVE_TRIM_FACTOR);
  nodeSetSocketAvailability(length_end_socket, mode == GEO_NODE_CURVE_TRIM_LENGTH);
}

namespace blender::nodes {

static void trim_spline(Spline &spline, const float length_start, const float length_end)
{
  BLI_assert(length_start <= length_end);

  std::cout << "\n TRIM SPLINE \n";
  std::cout << "Start length: " << length_start << "\n";
  std::cout << "End length: " << length_end << "\n";
  Span<float> lengths = spline.evaluated_lengths();
  const float *lower = std::lower_bound(lengths.begin(), lengths.end(), length_start);
  const float *upper = std::upper_bound(lengths.begin(), lengths.end(), length_end);
  const int i_lower = lower - lengths.begin();
  const int i_upper = upper - lengths.begin();
  std::cout << "i_lower: " << i_lower << "\n";
  std::cout << "i_upper: " << i_upper << "\n";
  BLI_assert(i_lower <= i_upper);

  Span<PointMapping> mappings = spline.evaluated_mappings();
  const int i_control_lower = mappings[i_lower].control_point_index;
  const int i_control_upper = mappings[i_upper].control_point_index;
  std::cout << "i_control_lower: " << i_control_lower << "\n";
  std::cout << "i_control_upper: " << i_control_upper << "\n";
  BLI_assert(i_control_lower <= i_control_upper);

}

static void geo_node_curve_trim_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");

  const bNode &node = params.node();
  const NodeGeometryCurveTrim &node_storage = *(const NodeGeometryCurveTrim *)node.storage;
  const GeometryNodeCurveTrimMode mode = (GeometryNodeCurveTrimMode)node_storage.mode;

  params.error_message_add(NodeWarningType::Info, "The node doesn't do anything yet");

  if (!geometry_set.has_curve()) {
    params.set_output("Geometry", geometry_set);
    return;
  }

  DCurve &curve = *geometry_set.get_curve_for_write();

  switch (mode) {
    case GEO_NODE_CURVE_TRIM_FACTOR: {
      const float factor_start = params.extract_input<float>("Start");
      const float factor_end = params.extract_input<float>("End");
      for (SplinePtr &spline : curve.splines) {
        const float length = spline->evaluated_lengths().last();
        const float start = factor_start * length;
        const float end = factor_end * length;
        trim_spline(*spline, std::min(start, end), std::max(start, end));
      }
      break;
    }
    case GEO_NODE_CURVE_TRIM_LENGTH: {
      const float start = params.extract_input<float>("Start_001");
      const float end = params.extract_input<float>("End_001");
      for (SplinePtr &spline : curve.splines) {
        trim_spline(*spline, std::min(start, end), std::max(start, end));
      }
      break;
    }
    default:
      BLI_assert_unreachable();
      break;
  }

  params.set_output("Geometry", geometry_set);
}

}  // namespace blender::nodes

void register_node_type_geo_curve_trim()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_CURVE_TRIM, "Curve Trim", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, geo_node_curve_trim_in, geo_node_curve_trim_out);
  node_type_init(&ntype, geo_node_curve_trim_init);
  node_type_update(&ntype, geo_node_curve_trim_update);
  node_type_storage(
      &ntype, "NodeGeometryCurveTrim", node_free_standard_storage, node_copy_standard_storage);

  ntype.geometry_node_execute = blender::nodes::geo_node_curve_trim_exec;
  ntype.draw_buttons = geo_node_curve_trim_layout;
  nodeRegisterType(&ntype);
}
