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

#include "UI_interface.h"
#include "UI_resources.h"

#include "FN_multi_function_builder.hh"

#include "node_geometry_util.hh"

namespace blender::nodes {

static void geo_node_point_translate_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Geometry");
  b.add_input<decl::String>("Translation");
  b.add_input<decl::Vector>("Translation", "Translation_001").subtype(PROP_TRANSLATION);
  b.add_input<decl::Bool>("Selection").default_value(true);
  b.add_output<decl::Geometry>("Geometry");
}

static void geo_node_point_translate_layout(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
{
  uiLayoutSetPropSep(layout, true);
  uiLayoutSetPropDecorate(layout, false);
  uiItemR(layout, ptr, "input_type", 0, IFACE_("Type"), ICON_NONE);
}

static void execute_on_component_legacy(GeoNodeExecParams params, GeometryComponent &component)
{
  OutputAttribute_Typed<float3> position_attribute =
      component.attribute_try_get_for_output<float3>("position", ATTR_DOMAIN_POINT, {0, 0, 0});
  if (!position_attribute) {
    return;
  }
  GVArray_Typed<float3> attribute = params.get_input_attribute<float3>(
      "Translation", component, ATTR_DOMAIN_POINT, {0, 0, 0});

  for (const int i : IndexRange(attribute.size())) {
    position_attribute->set(i, position_attribute->get(i) + attribute[i]);
  }

  position_attribute.save();
}

class SpanFieldInput final : public fn::FieldInput {
  GSpan span_;

 public:
  SpanFieldInput(GSpan span) : FieldInput(CPPType::get<float3>(), "Span"), span_(span)
  {
  }
  const GVArray *get_varray_for_context(const fn::FieldContext &UNUSED(context),
                                        IndexMask UNUSED(mask),
                                        ResourceScope &scope) const final
  {
    return &scope.construct<fn::GVArray_For_GSpan>(__func__, span_);
  }
};

static void execute_on_component(GeometryComponent &component,
                                 const Field<bool> &selection_field,
                                 const Field<float3> &translation_field)
{
  GeometryComponentFieldContext field_context{component, ATTR_DOMAIN_POINT};
  const int domain_size = component.attribute_domain_size(ATTR_DOMAIN_POINT);

  fn::FieldEvaluator selection_evaluator{field_context, domain_size};
  selection_evaluator.add(selection_field);
  selection_evaluator.evaluate();
  const IndexMask selection = selection_evaluator.get_evaluated_as_mask(0);

  OutputAttribute_Typed<float3> positions = component.attribute_try_get_for_output<float3>(
      "position", ATTR_DOMAIN_POINT, {0, 0, 0});
  MutableSpan<float3> position_span = positions.as_span();
  fn::Field<float3> position_field{std::make_shared<SpanFieldInput>(position_span.as_span())};

  /* Add an add operation field on top of the provided translation field, which can be evaluated
   * directly into the position virtual array. That way, any optimizations can be done more
   * generally for the whole evaluation system. In the general case it may not work to share the
   * same span for the input and output of an evaluation, but in this case there there is only one
   * output, so it is fine. */
  static const fn::CustomMF_SI_SI_SO<float3, float3, float3> add_fn = {
      "Add", [](float3 a, float3 b) { return a + b; }};
  std::shared_ptr<fn::FieldOperation> add_operation = std::make_shared<fn::FieldOperation>(
      fn::FieldOperation(add_fn, {position_field, translation_field}));

  fn::FieldEvaluator position_evaluator{field_context, &selection};
  position_evaluator.add_with_destination({add_operation}, position_span);
  position_evaluator.evaluate();

  positions.save();
}

static void geo_node_point_translate_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");

  geometry_set = geometry_set_realize_instances(geometry_set);

  const NodeGeometryPointTranslate &storage =
      *(const NodeGeometryPointTranslate *)params.node().storage;

  static const Array<GeometryComponentType> types{
      GEO_COMPONENT_TYPE_MESH,
      GEO_COMPONENT_TYPE_POINT_CLOUD,
      GEO_COMPONENT_TYPE_CURVE,
  };

  /* TODO: Remove legacy string input and add versioning. */
  if (storage.input_type == GEO_NODE_ATTRIBUTE_INPUT_ATTRIBUTE) {
    for (const GeometryComponentType type : types) {
      if (geometry_set.has(type)) {
        execute_on_component_legacy(params, geometry_set.get_component_for_write(type));
      }
    }
    params.error_message_add(NodeWarningType::Info, "Selection not supported in legacy mode");
  }
  else {
    Field<bool> selection = params.extract_input<Field<bool>>("Selection");
    Field<float3> translation = params.extract_input<Field<float3>>("Translation_001");
    for (const GeometryComponentType type : types) {
      if (geometry_set.has(type)) {
        execute_on_component(geometry_set.get_component_for_write(type), selection, translation);
      }
    }
  }

  params.set_output("Geometry", std::move(geometry_set));
}

static void geo_node_point_translate_init(bNodeTree *UNUSED(tree), bNode *node)
{
  NodeGeometryPointTranslate *data = (NodeGeometryPointTranslate *)MEM_callocN(
      sizeof(NodeGeometryPointTranslate), __func__);

  data->input_type = GEO_NODE_ATTRIBUTE_INPUT_VECTOR;
  node->storage = data;
}

static void geo_node_point_translate_update(bNodeTree *UNUSED(ntree), bNode *node)
{
  NodeGeometryPointTranslate &node_storage = *(NodeGeometryPointTranslate *)node->storage;

  update_attribute_input_socket_availabilities(
      *node, "Translation", (GeometryNodeAttributeInputMode)node_storage.input_type);
}

}  // namespace blender::nodes

void register_node_type_geo_point_translate()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_POINT_TRANSLATE, "Point Translate", NODE_CLASS_GEOMETRY, 0);
  node_type_init(&ntype, blender::nodes::geo_node_point_translate_init);
  node_type_update(&ntype, blender::nodes::geo_node_point_translate_update);
  node_type_storage(&ntype,
                    "NodeGeometryPointTranslate",
                    node_free_standard_storage,
                    node_copy_standard_storage);
  ntype.declare = blender::nodes::geo_node_point_translate_declare;
  ntype.geometry_node_execute = blender::nodes::geo_node_point_translate_exec;
  ntype.draw_buttons = blender::nodes::geo_node_point_translate_layout;
  nodeRegisterType(&ntype);
}
