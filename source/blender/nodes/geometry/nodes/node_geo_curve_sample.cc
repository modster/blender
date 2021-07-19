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

static bNodeSocketTemplate geo_node_curve_sample_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_STRING, N_("Parameter")},
    {SOCK_GEOMETRY, N_("Curve")},
    {SOCK_STRING, N_("Position")},
    {SOCK_STRING, N_("Tangent")},
    {SOCK_STRING, N_("Normal")},
    {SOCK_STRING, N_("Attribute")},
    {SOCK_STRING, N_("Result")},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_curve_sample_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

static void geo_node_curve_sample_layout(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
{
  uiItemR(layout, ptr, "mode", UI_ITEM_R_EXPAND, nullptr, ICON_NONE);
}

static void geo_node_curve_sample_init(bNodeTree *UNUSED(tree), bNode *node)
{
  NodeGeometryCurveSample *data = (NodeGeometryCurveSample *)MEM_callocN(
      sizeof(NodeGeometryCurveSample), __func__);

  data->mode = GEO_NODE_CURVE_INTERPOLATE_FACTOR;
  node->storage = data;
}

static void geo_node_curve_sample_update(bNodeTree *UNUSED(ntree), bNode *node)
{
  const NodeGeometryCurveSample &node_storage = *(NodeGeometryCurveSample *)node->storage;
  const GeometryNodeCurveInterpolateMode mode = (GeometryNodeCurveInterpolateMode)
                                                    node_storage.mode;

  bNodeSocket *parameter_socket = ((bNodeSocket *)node->inputs.first)->next;

  if (mode == GEO_NODE_CURVE_INTERPOLATE_FACTOR) {
    node_sock_label(parameter_socket, "Factor");
  }
  else {
    BLI_assert(mode == GEO_NODE_CURVE_INTERPOLATE_LENGTH);
    node_sock_label(parameter_socket, "Length");
  }
}

namespace blender::nodes {

static AttributeDomain get_result_domain(const GeometryComponent &component,
                                         const StringRef parameter_name,
                                         const StringRef result_name)
{
  std::optional<AttributeMetaData> result_info = component.attribute_get_meta_data(result_name);
  if (result_info) {
    return result_info->domain;
  }
  std::optional<AttributeMetaData> parameter_info = component.attribute_get_meta_data(
      parameter_name);
  if (parameter_info) {
    return parameter_info->domain;
  }
  return ATTR_DOMAIN_POINT;
}

static CustomDataType get_result_type(const CurveComponent &curve_component,
                                      const StringRef attribute_name)
{
  std::optional<AttributeMetaData> curve_meta_data = curve_component.attribute_get_meta_data(
      attribute_name);
  return curve_meta_data->data_type;
}

struct SamplePair {
  GSpan src;
  GMutableSpan dst;
};

static void sample_with_lookups(const SamplePair &sample, Span<Spline::LookupResult> lookups)
{
  blender::attribute_math::convert_to_static_type(sample.src.type(), [&](auto dummy) {
    using T = decltype(dummy);
    Span<T> src = sample.src.typed<T>();
    MutableSpan<T> dst = sample.dst.typed<T>();

    threading::parallel_for(lookups.index_range(), 1024, [&](IndexRange range) {
      for (const int i : range) {
        dst[i] = attribute_math::mix2<T>(lookups[i].factor,
                                         src[lookups[i].evaluated_index],
                                         src[lookups[i].next_evaluated_index]);
      }
    });
  });
}

static void execute_on_component(GeometryComponent &component,
                                 const CurveComponent &curve_component,
                                 const GeometryNodeCurveInterpolateMode mode,
                                 const StringRef pararameter_name,
                                 const StringRef position_name,
                                 const StringRef tangent_name,
                                 const StringRef normal_name,
                                 const StringRef attribute_name,
                                 const StringRef result_name)
{
  const CurveEval &curve = *curve_component.get_for_read();
  const Spline &spline = *curve.splines().first();
  const float length = spline.length();

  const AttributeDomain domain = get_result_domain(component, pararameter_name, result_name);

  GVArray_Typed<float> parameters = component.attribute_get_for_read<float>(
      pararameter_name, domain, 0.0f);

  Array<Spline::LookupResult> lookups(parameters.size());
  if (mode == GEO_NODE_CURVE_INTERPOLATE_LENGTH) {
    threading::parallel_for(lookups.index_range(), 1024, [&](IndexRange range) {
      for (const int i : range) {
        lookups[i] = spline.lookup_evaluated_length(std::clamp(parameters[i], 0.0f, length));
      }
    });
  }
  else {
    threading::parallel_for(lookups.index_range(), 1024, [&](IndexRange range) {
      for (const int i : range) {
        lookups[i] = spline.lookup_evaluated_factor(std::clamp(parameters[i], 0.0f, 1.0f));
      }
    });
  }

  if (!position_name.is_empty()) {
    OutputAttribute positions = component.attribute_try_get_for_output_only(
        position_name, domain, CD_PROP_FLOAT3);
    if (positions) {
      sample_with_lookups({spline.evaluated_positions(), positions.as_span()}, lookups);
      positions.save();
    }
  }
  if (!tangent_name.is_empty()) {
    OutputAttribute tangents = component.attribute_try_get_for_output_only(
        tangent_name, domain, CD_PROP_FLOAT3);
    if (tangents) {
      sample_with_lookups({spline.evaluated_tangents(), tangents.as_span()}, lookups);
      tangents.save();
    }
  }
  if (!normal_name.is_empty()) {
    OutputAttribute normals = component.attribute_try_get_for_output_only(
        normal_name, domain, CD_PROP_FLOAT3);
    if (normals) {
      sample_with_lookups({spline.evaluated_normals(), normals.as_span()}, lookups);
      normals.save();
    }
  }
  if (!attribute_name.is_empty() && !result_name.is_empty()) {
    OutputAttribute dst = component.attribute_try_get_for_output_only(
        result_name, domain, get_result_type(curve_component, attribute_name));
    if (dst) {
      std::optional<GSpan> src = spline.attributes.get_for_read(attribute_name);
      if (src) {
        GVArrayPtr attribute_interpolated = spline.interpolate_to_evaluated(*src);
        sample_with_lookups({attribute_interpolated->get_internal_span(), dst.as_span()}, lookups);
        dst.save();
      }
    }
  }
}

static void geo_node_sample_exec(GeoNodeExecParams params)
{
  const NodeGeometryCurveSample &node_storage = *(NodeGeometryCurveSample *)params.node().storage;
  const GeometryNodeCurveInterpolateMode mode = (GeometryNodeCurveInterpolateMode)
                                                    node_storage.mode;

  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");
  GeometrySet curve_set = params.extract_input<GeometrySet>("Curve");

  geometry_set = bke::geometry_set_realize_instances(geometry_set);
  curve_set = bke::geometry_set_realize_instances(curve_set);

  if (!curve_set.has_curve()) {
    params.set_output("Geometry", geometry_set);
    return;
  }

  const std::string pararameter_name = params.extract_input<std::string>("Parameter");
  const std::string position_name = params.extract_input<std::string>("Position");
  const std::string tangent_name = params.extract_input<std::string>("Tangent");
  const std::string normal_name = params.extract_input<std::string>("Normal");
  const std::string attribute_name = params.extract_input<std::string>("Attribute");
  const std::string result_name = params.extract_input<std::string>("Result");

  const CurveComponent &curve_component = *curve_set.get_component_for_read<CurveComponent>();

  if (attribute_name.empty()) {
    if (position_name.empty() && tangent_name.empty() && normal_name.empty()) {
      params.set_output("Geometry", geometry_set);
      return;
    }
  }
  else if (!curve_component.attribute_exists(attribute_name)) {
    params.error_message_add(NodeWarningType::Error,
                             TIP_("No attribute with name \"") + attribute_name + "\"");
  }

  for (const GeometryComponentType type :
       {GEO_COMPONENT_TYPE_MESH, GEO_COMPONENT_TYPE_POINT_CLOUD, GEO_COMPONENT_TYPE_CURVE}) {
    if (geometry_set.has(type)) {
      execute_on_component(geometry_set.get_component_for_write(type),
                           curve_component,
                           mode,
                           pararameter_name,
                           position_name,
                           tangent_name,
                           normal_name,
                           attribute_name,
                           result_name);
    }
  }

  params.set_output("Geometry", std::move(geometry_set));
}

}  // namespace blender::nodes

void register_node_type_geo_curve_sample()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_CURVE_SAMPLE, "Sample Curve", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, geo_node_curve_sample_in, geo_node_curve_sample_out);
  ntype.draw_buttons = geo_node_curve_sample_layout;
  node_type_storage(
      &ntype, "NodeGeometryCurveSample", node_free_standard_storage, node_copy_standard_storage);
  node_type_init(&ntype, geo_node_curve_sample_init);
  node_type_update(&ntype, geo_node_curve_sample_update);
  ntype.geometry_node_execute = blender::nodes::geo_node_sample_exec;
  nodeRegisterType(&ntype);
}
