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

#include "node_geometry_util.hh"

static bNodeSocketTemplate geo_node_point_translate_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_VECTOR,
     N_("Translation"),
     0.0f,
     0.0f,
     0.0f,
     1.0f,
     -FLT_MAX,
     FLT_MAX,
     PROP_TRANSLATION,
     SOCK_FIELD},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_point_translate_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

namespace blender::nodes {

static void execute_on_component(GeoNodeExecParams params, GeometryComponent &component)
{
  OutputAttribute_Typed<float3> position_attribute =
      component.attribute_try_get_for_output<float3>("position", ATTR_DOMAIN_POINT, {0, 0, 0});
  if (!position_attribute) {
    return;
  }

  bke::FieldRef<float3> field = params.get_input_field<float3>("Translation");
  bke::FieldInputs field_inputs = field->prepare_inputs();
  Vector<std::unique_ptr<bke::FieldInputValue>> field_input_values;
  prepare_field_inputs(field_inputs, component, ATTR_DOMAIN_POINT, field_input_values);
  bke::FieldOutput field_output = field->evaluate(
      IndexRange(component.attribute_domain_size(ATTR_DOMAIN_POINT)), field_inputs);

  GVArray_Typed<float3> translation{field_output.varray_ref()};

  for (const int i : IndexRange(translation.size())) {
    position_attribute->set(i, position_attribute->get(i) + translation[i]);
  }

  position_attribute.save();
}

static void geo_node_point_translate_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");

  geometry_set = geometry_set_realize_instances(geometry_set);

  if (geometry_set.has<MeshComponent>()) {
    execute_on_component(params, geometry_set.get_component_for_write<MeshComponent>());
  }
  if (geometry_set.has<PointCloudComponent>()) {
    execute_on_component(params, geometry_set.get_component_for_write<PointCloudComponent>());
  }
  if (geometry_set.has<CurveComponent>()) {
    execute_on_component(params, geometry_set.get_component_for_write<CurveComponent>());
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
  node_type_socket_templates(&ntype, geo_node_point_translate_in, geo_node_point_translate_out);
  node_type_init(&ntype, blender::nodes::geo_node_point_translate_init);
  node_type_update(&ntype, blender::nodes::geo_node_point_translate_update);
  node_type_storage(&ntype,
                    "NodeGeometryPointTranslate",
                    node_free_standard_storage,
                    node_copy_standard_storage);
  ntype.geometry_node_execute = blender::nodes::geo_node_point_translate_exec;
  nodeRegisterType(&ntype);
}
