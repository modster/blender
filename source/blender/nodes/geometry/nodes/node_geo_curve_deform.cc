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
#include "BKE_geometry_set_instances.hh"
#include "BKE_spline.hh"

#include "UI_interface.h"
#include "UI_resources.h"

#include "node_geometry_util.hh"

using blender::bke::GeometryInstanceGroup;

static bNodeSocketTemplate geo_node_curve_deform_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_GEOMETRY, N_("Curve")},
    {SOCK_STRING, N_("Parameter")},
    {SOCK_FLOAT, N_("Parameter"), 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, FLT_MAX},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_curve_deform_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

static void geo_node_curve_deform_layout(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
{
  const bNode *node = (bNode *)ptr->data;
  NodeGeometryCurveDeform &node_storage = *(NodeGeometryCurveDeform *)node->storage;
  const GeometryNodeCurveDeformMode mode = (GeometryNodeCurveDeformMode)node_storage.input_mode;

  uiItemR(layout, ptr, "input_mode", UI_ITEM_R_EXPAND, nullptr, ICON_NONE);

  if (mode == GEO_NODE_CURVE_DEFORM_POSITION) {
    uiItemR(layout, ptr, "position_axis", UI_ITEM_R_EXPAND, nullptr, ICON_NONE);
  }
  else {
    uiItemR(layout, ptr, "attribute_input_type", UI_ITEM_R_EXPAND, nullptr, ICON_NONE);
  }
}

static void geo_node_curve_deform_init(bNodeTree *UNUSED(tree), bNode *node)
{
  NodeGeometryCurveDeform *data = (NodeGeometryCurveDeform *)MEM_callocN(
      sizeof(NodeGeometryCurveDeform), __func__);

  data->input_mode = GEO_NODE_CURVE_DEFORM_POSITION;
  data->position_axis = GEO_NODE_CURVE_DEFORM_POSX;
  data->attribute_input_type = GEO_NODE_ATTRIBUTE_INPUT_ATTRIBUTE;
  node->storage = data;
}

namespace blender::nodes {

static void geo_node_curve_deform_update(bNodeTree *UNUSED(ntree), bNode *node)
{
  NodeGeometryCurveDeform &node_storage = *(NodeGeometryCurveDeform *)node->storage;
  const GeometryNodeCurveDeformMode mode = (GeometryNodeCurveDeformMode)node_storage.input_mode;

  bNodeSocket *attribute_socket = ((bNodeSocket *)node->inputs.first)->next->next;

  nodeSetSocketAvailability(attribute_socket, mode == GEO_NODE_CURVE_DEFORM_ATTRIBUTE);
  if (mode == GEO_NODE_CURVE_DEFORM_ATTRIBUTE) {
    update_attribute_input_socket_availabilities(
        *node, "Parameter", (GeometryNodeAttributeInputMode)node_storage.attribute_input_type);
  }
}

static void execute_on_component(GeoNodeExecParams params,
                                 GeometryComponent &component,
                                 const CurveEval &curve)
{
}

static void geo_node_curve_deform_exec(GeoNodeExecParams params)
{
  GeometrySet deform_geometry_set = params.extract_input<GeometrySet>("Geometry");
  GeometrySet curve_geometry_set = params.extract_input<GeometrySet>("Curve");

  deform_geometry_set = bke::geometry_set_realize_instances(deform_geometry_set);
  curve_geometry_set = bke::geometry_set_realize_instances(curve_geometry_set);

  const CurveEval *curve = curve_geometry_set.get_curve_for_read();
  if (curve == nullptr) {
    params.error_message_add(NodeWarningType::Error, TIP_("Curve input must contain curve data"));
  }

  if (deform_geometry_set.has<MeshComponent>()) {
    execute_on_component(
        params, deform_geometry_set.get_component_for_write<MeshComponent>(), *curve);
  }
  if (deform_geometry_set.has<PointCloudComponent>()) {
    execute_on_component(
        params, deform_geometry_set.get_component_for_write<PointCloudComponent>(), *curve);
  }
  if (deform_geometry_set.has<CurveComponent>()) {
    execute_on_component(
        params, deform_geometry_set.get_component_for_write<CurveComponent>(), *curve);
  }

  params.set_output("Geometry", GeometrySet());
}

}  // namespace blender::nodes

void register_node_type_geo_curve_deform()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_CURVE_DEFORM, "Curve Deform", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, geo_node_curve_deform_in, geo_node_curve_deform_out);
  ntype.draw_buttons = geo_node_curve_deform_layout;
  node_type_storage(
      &ntype, "NodeGeometryCurveDeform", node_free_standard_storage, node_copy_standard_storage);
  node_type_init(&ntype, geo_node_curve_deform_init);
  node_type_update(&ntype, blender::nodes::geo_node_curve_deform_update);
  ntype.geometry_node_execute = blender::nodes::geo_node_curve_deform_exec;
  nodeRegisterType(&ntype);
}
