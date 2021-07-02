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

#include "BLI_array.hh"
#include "BLI_task.hh"
#include "BLI_timeit.hh"

#include "BKE_attribute_math.hh"
#include "BKE_spline.hh"

#include "UI_interface.h"
#include "UI_resources.h"

#include "node_geometry_util.hh"

using blender::fn::GVArray_For_GSpan;
using blender::fn::GVArray_For_Span;
using blender::fn::GVArray_Typed;

static bNodeSocketTemplate geo_node_curve_resample_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_STRING, N_("Parameter")},
    {SOCK_GEOMETRY, N_("Curve")},
    {SOCK_GEOMETRY, N_("Attribute")},
    {SOCK_GEOMETRY, N_("Result")},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_curve_resample_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

static void geo_node_curve_resample_layout(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
{
  uiItemR(layout, ptr, "mode", UI_ITEM_R_EXPAND, nullptr, ICON_NONE);
}

static void geo_node_curve_resample_init(bNodeTree *UNUSED(tree), bNode *node)
{
  NodeGeometryCurveSample *data = (NodeGeometryCurveSample *)MEM_callocN(
      sizeof(NodeGeometryCurveSample), __func__);

  data->mode = GEO_NODE_CURVE_RESAMPLE_COUNT;
  node->storage = data;
}

static void geo_node_curve_resample_update(bNodeTree *UNUSED(ntree), bNode *node)
{
  const NodeGeometryCurveSample &node_storage = *(NodeGeometryCurveSample *)node->storage;
  const GeometryNodeCurveSampleMode mode = (GeometryNodeCurveSampleMode)node_storage.mode;

  bNodeSocket *parameter_socket = ((bNodeSocket *)node->inputs.first)->next;

  if (mode == GEO_NODE_CURVE_SAMPLE_FACTOR) {
    node_sock_label(parameter_socket, "Factor");
  }
  else {
    BLI_assert(mode == GEO_NODE_CURVE_SAMPLE_LENGTH);
    node_sock_label(parameter_socket, "Length");
  }
}

namespace blender::nodes {

static void execute_on_component(GeometryComponent &component,
                                 const CurveComponent &curve_component,
                                 const StringRef pararameter_name,
                                 const StringRef attribute_name,
                                 const StringRef result_name)
{
  GVArray_Typed<float> parameters = component.attribute_get_for_read<float>(pararameter_name,
                                                                            0.0f);
  ReadAttributeLookup attribute = curve_component.attribute_try_get_for_read(attribute_name,
                                                                             ATTR_DOMAIN_POINT);
}

static void geo_node_resample_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");
  GeometrySet curve_set = params.extract_input<GeometrySet>("Curve");

  geometry_set = bke::geometry_set_realize_instances(geometry_set);
  curve_set = bke::geometry_set_realize_instances(curve_set);

  if (!curve_set.has_curve()) {
    params.set_output("Geometry", geometry_set);
    return;
  }

  const std::string pararameter_name = params.extract_input<std::string>("Parameter");
  const std::string attribute_name = params.extract_input<std::string>("Attribute");
  const std::string result_name = params.extract_input<std::string>("Result");
  if (pararameter_name.empty() || attribute_name.empty() || result_name.empty()) {
    params.set_output("Geometry", geometry_set);
    return;
  }

  const CurveComponent &curve_component = *curve_set.get_component_for_read<CurveComponent>();
  const CurveEval &curve = *curve_component.get_for_read();

  if (!curve_component.attribute_exists(attribute_name)) {
    params.error_message_add(NodeWarningType::Error,
                             TIP_("No attribute with name \"") + attribute_name + "\"");
    params.set_output("Geometry", geometry_set);
    return;
  }

  for (const GeometryComponentType type :
       {GEO_COMPONENT_TYPE_MESH, GEO_COMPONENT_TYPE_POINT_CLOUD, GEO_COMPONENT_TYPE_CURVE}) {
    if (geometry_set.has(type)) {
      execute_on_component(geometry_set.get_component_for_write(type),
                           curve_component,
                           pararameter_name,
                           attribute_name,
                           result_name);
    }
  }

  params.set_output("Geometry", std::move(geometry_set));
}

}  // namespace blender::nodes

void register_node_type_geo_curve_resample()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_CURVE_RESAMPLE, "Resample Curve", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, geo_node_curve_resample_in, geo_node_curve_resample_out);
  ntype.draw_buttons = geo_node_curve_resample_layout;
  node_type_storage(
      &ntype, "NodeGeometryCurveSample", node_free_standard_storage, node_copy_standard_storage);
  node_type_init(&ntype, geo_node_curve_resample_init);
  node_type_update(&ntype, geo_node_curve_resample_update);
  ntype.geometry_node_execute = blender::nodes::geo_node_resample_exec;
  nodeRegisterType(&ntype);
}
