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

static bNodeSocketTemplate geo_node_set_position_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_VECTOR,
     N_("Position"),
     0.0f,
     0.0f,
     0.0f,
     1.0f,
     -FLT_MAX,
     FLT_MAX,
     PROP_TRANSLATION,
     SOCK_FIELD},
    {SOCK_BOOLEAN, N_("Selection"), 1, 0, 0, 0, 0, 0, PROP_NONE, SOCK_HIDE_VALUE | SOCK_FIELD},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_set_position_out[] = {
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

  bke::FieldRef<bool> selection_field = params.get_input_field<bool>("Selection");
  bke::FieldInputs selection_field_inputs = selection_field->prepare_inputs();
  Vector<std::unique_ptr<bke::FieldInputValue>> selection_field_input_values;
  prepare_field_inputs(
      selection_field_inputs, component, ATTR_DOMAIN_POINT, selection_field_input_values);
  bke::FieldOutput selection_field_output = selection_field->evaluate(
      IndexRange(component.attribute_domain_size(ATTR_DOMAIN_POINT)), selection_field_inputs);
  GVArray_Typed<bool> selection{selection_field_output.varray_ref()};

  bke::FieldRef<float3> field = params.get_input_field<float3>("Position");
  bke::FieldInputs field_inputs = field->prepare_inputs();
  Vector<std::unique_ptr<bke::FieldInputValue>> field_input_values;
  prepare_field_inputs(field_inputs, component, ATTR_DOMAIN_POINT, field_input_values);
  bke::FieldOutput field_output = field->evaluate(
      IndexRange(component.attribute_domain_size(ATTR_DOMAIN_POINT)), field_inputs);

  GVArray_Typed<float3> new_positions{field_output.varray_ref()};

  for (const int i : IndexRange(new_positions.size())) {
    if (selection[i]) {
      position_attribute->set(i, new_positions[i]);
    }
  }

  position_attribute.save();
}

static void geo_node_set_position_exec(GeoNodeExecParams params)
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

}  // namespace blender::nodes

void register_node_type_geo_set_position()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_SET_POSITION, "Set Position", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, geo_node_set_position_in, geo_node_set_position_out);
  ntype.geometry_node_execute = blender::nodes::geo_node_set_position_exec;
  nodeRegisterType(&ntype);
}
