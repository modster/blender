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
#include "BLI_float4x4.hh"
#include "BLI_task.hh"
#include "BLI_timeit.hh"

#include "BKE_attribute_math.hh"
#include "BKE_geometry_set_instances.hh"
#include "BKE_spline.hh"

#include "UI_interface.h"
#include "UI_resources.h"

#include "node_geometry_util.hh"

using blender::float4x4;
using blender::bke::GeometryInstanceGroup;

static bNodeSocketTemplate geo_node_curve_deform_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_GEOMETRY, N_("Curve")},
    {SOCK_STRING, N_("Factor")},
    {SOCK_FLOAT, N_("Factor"), 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, FLT_MAX},
    // {SOCK_BOOLEAN, N_("Use Radius")},
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

  uiItemR(layout, ptr, "position_axis", UI_ITEM_R_EXPAND, nullptr, ICON_NONE);

  uiItemR(layout, ptr, "input_mode", UI_ITEM_R_EXPAND, nullptr, ICON_NONE);

  if (mode == GEO_NODE_CURVE_DEFORM_ATTRIBUTE) {
    uiLayoutSetPropSep(layout, true);
    uiLayoutSetPropDecorate(layout, false);
    uiItemR(layout, ptr, "attribute_input_type", 0, IFACE_("Factor"), ICON_NONE);
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
  update_attribute_input_socket_availabilities(
      *node,
      "Factor",
      (GeometryNodeAttributeInputMode)node_storage.attribute_input_type,
      mode == GEO_NODE_CURVE_DEFORM_ATTRIBUTE);
}

static void spline_deform(const Spline &spline,
                          Span<float> index_factors,
                          const int axis_index,
                          MutableSpan<float3> positions)
{
  Span<float3> spline_positions = spline.evaluated_positions();
  Span<float3> spline_tangents = spline.evaluated_tangents();
  Span<float3> spline_normals = spline.evaluated_normals();
  GVArray_Typed<float> radii = spline.interpolate_to_evaluated_points(spline.radii());

  parallel_for(positions.index_range(), 1024, [&](IndexRange range) {
    for (const int i : range) {
      const Spline::LookupResult interp = spline.lookup_data_from_index_factor(index_factors[i]);
      const int index = interp.evaluated_index;
      const int next_index = interp.next_evaluated_index;
      const float factor = interp.factor;

      float4x4 matrix = float4x4::from_normalized_axis_data(
          {0, 0, 0},
          float3::interpolate(spline_normals[index], spline_normals[next_index], factor)
              .normalized(),
          float3::interpolate(spline_tangents[index], spline_tangents[next_index], factor)
              .normalized());
      matrix.apply_scale(interpf(radii[next_index], radii[index], factor));

      float3 position = positions[i];
      position[axis_index] = 0.0f;

      positions[i] = matrix * position;
      positions[i] += float3::interpolate(
          spline_positions[index], spline_positions[next_index], factor);
    }
  });
}

static int axis_to_index(const GeometryNodeCurveDeformPositionAxis axis)
{
  switch (axis) {
    case GEO_NODE_CURVE_DEFORM_POSX:
    case GEO_NODE_CURVE_DEFORM_NEGX:
      return 0;
    case GEO_NODE_CURVE_DEFORM_POSY:
    case GEO_NODE_CURVE_DEFORM_NEGY:
      return 1;
    case GEO_NODE_CURVE_DEFORM_POSZ:
    case GEO_NODE_CURVE_DEFORM_NEGZ:
      return 2;
  }
  BLI_assert_unreachable();
  return -1;
}

static bool axis_is_negative(const GeometryNodeCurveDeformPositionAxis axis)
{
  switch (axis) {
    case GEO_NODE_CURVE_DEFORM_POSX:
    case GEO_NODE_CURVE_DEFORM_POSY:
    case GEO_NODE_CURVE_DEFORM_POSZ:
      return false;
    case GEO_NODE_CURVE_DEFORM_NEGX:
    case GEO_NODE_CURVE_DEFORM_NEGY:
    case GEO_NODE_CURVE_DEFORM_NEGZ:
      return true;
  }
  BLI_assert_unreachable();
  return false;
}

static float find_upper_bound(Span<float3> positions, const int index)
{
  float max = FLT_MIN;
  for (const float3 &position : positions) {
    if (position[index] > max) {
      max = position[index];
    }
  }
  return max;
}

static float find_lower_bound(Span<float3> positions, const int index)
{
  float min = FLT_MAX;
  for (const float3 &position : positions) {
    if (position[index] < min) {
      min = position[index];
    }
  }
  return min;
}

static void execute_on_component(const GeoNodeExecParams &params,
                                 const CurveEval &curve,
                                 GeometryComponent &component)
{
  const NodeGeometryCurveDeform &node_storage = *(NodeGeometryCurveDeform *)params.node().storage;
  const GeometryNodeCurveDeformMode mode = (GeometryNodeCurveDeformMode)node_storage.input_mode;
  const GeometryNodeCurveDeformPositionAxis axis = (GeometryNodeCurveDeformPositionAxis)
                                                       node_storage.position_axis;
  const int axis_index = axis_to_index(axis);

  if (curve.splines().size() == 0) {
    BLI_assert_unreachable();
    return;
  }

  const Spline &spline = *curve.splines().first();
  const float total_length = spline.length();

  const int size = component.attribute_domain_size(ATTR_DOMAIN_POINT);
  OutputAttribute_Typed<float3> position_attribute =
      component.attribute_try_get_for_output<float3>("position", ATTR_DOMAIN_POINT, {0, 0, 0});
  MutableSpan<float3> positions = position_attribute.as_span();

  Array<float> parameters(size);

  if (mode == GEO_NODE_CURVE_DEFORM_POSITION) {
    if (axis_is_negative(axis)) {
      parallel_for(positions.index_range(), 4096, [&](IndexRange range) {
        for (const int i : range) {
          parameters[i] = std::clamp(positions[i][axis_index], 0.0f, total_length);
        }
      });
    }
    else {
      parallel_for(positions.index_range(), 4096, [&](IndexRange range) {
        for (const int i : range) {
          parameters[i] = total_length - std::clamp(positions[i][axis_index], 0.0f, total_length);
        }
      });
    }
  }
  else {
    BLI_assert(mode == GEO_NODE_CURVE_DEFORM_ATTRIBUTE);
    GVArrayPtr attribute = params.get_input_attribute(
        "Factor", component, ATTR_DOMAIN_POINT, CD_PROP_FLOAT, nullptr);
    if (!attribute) {
      return;
    }

    GVArray_Typed<float> parameter_attribute{*attribute};
    for (const int i : IndexRange(size)) {
      parameters[i] = std::clamp(parameter_attribute[i], 0.0f, total_length);
    }
  }

  spline.sample_length_parameters_to_index_factors(parameters);
  spline_deform(spline, parameters, axis_index, positions);

  position_attribute.save();
}

static void geo_node_curve_deform_exec(GeoNodeExecParams params)
{
  GeometrySet deform_geometry_set = params.extract_input<GeometrySet>("Geometry");
  GeometrySet curve_geometry_set = params.extract_input<GeometrySet>("Curve");

  deform_geometry_set = bke::geometry_set_realize_instances(deform_geometry_set);

  /* TODO: Theoretically this could be easily avoided. */
  curve_geometry_set = bke::geometry_set_realize_instances(curve_geometry_set);

  const CurveEval *curve = curve_geometry_set.get_curve_for_read();
  if (curve == nullptr) {
    params.set_output("Geometry", GeometrySet());
    return;
  }

  if (deform_geometry_set.has<MeshComponent>()) {
    execute_on_component(
        params, *curve, deform_geometry_set.get_component_for_write<MeshComponent>());
  }
  if (deform_geometry_set.has<PointCloudComponent>()) {
    execute_on_component(
        params, *curve, deform_geometry_set.get_component_for_write<PointCloudComponent>());
  }
  if (deform_geometry_set.has<CurveComponent>()) {
    execute_on_component(
        params, *curve, deform_geometry_set.get_component_for_write<CurveComponent>());
  }

  params.set_output("Geometry", deform_geometry_set);
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
