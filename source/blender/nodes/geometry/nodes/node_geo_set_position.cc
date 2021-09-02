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

#include "DEG_depsgraph_query.h"

#include "node_geometry_util.hh"

namespace blender::nodes {

static void geo_node_set_position_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Geometry");
  b.add_input<decl::Vector>("Position");
  b.add_input<decl::Bool>("Selection").default_value(true);
  b.add_output<decl::Geometry>("Geometry");
}

static IndexMask index_mask_from_selection_varray(const VArray<bool> &selection,
                                                  Vector<int64_t> &r_indices)
{
  if (selection.is_single()) {
    if (selection.get_internal_single()) {
      return IndexRange(selection.size());
    }
    return IndexRange(0);
  }
  if (selection.is_span()) {
    Span<bool> selection_span = selection.get_internal_span();
    for (const int i : selection_span.index_range()) {
      if (selection_span[i]) {
        r_indices.append(i);
      }
    }
  }
  else {
    for (const int i : selection.index_range()) {
      if (selection[i]) {
        r_indices.append(i);
      }
    }
  }
  return r_indices.as_span();
}

static void try_set_position_in_component(GeometrySet &geometry_set,
                                          const GeometryComponentType component_type,
                                          const Field<bool> &selection_field,
                                          const Field<float3> &positions_field)
{
  if (!geometry_set.has(component_type)) {
    return;
  }
  GeometryComponent &component = geometry_set.get_component_for_write(component_type);
  GeometryComponentFieldContext field_context{component, ATTR_DOMAIN_POINT};
  const int domain_size = component.attribute_domain_size(ATTR_DOMAIN_POINT);
  const IndexMask full_mask{IndexRange(domain_size)};

  fn::FieldEvaluator selection_evaluator{field_context, &full_mask};
  const VArray<bool> *selection = nullptr;
  selection_evaluator.add(selection_field, &selection);
  selection_evaluator.evaluate();

  Vector<int64_t> mask_indices;
  const IndexMask selected_mask = index_mask_from_selection_varray(*selection, mask_indices);

  OutputAttribute_Typed<float3> position_attribute =
      component.attribute_try_get_for_output<float3>("position", ATTR_DOMAIN_POINT, {0, 0, 0});
  fn::FieldEvaluator position_evaluator{field_context, &selected_mask};
  position_evaluator.add_with_destination(positions_field, position_attribute.varray());
  position_evaluator.evaluate();
  position_attribute.save();
}

static void geo_node_set_position_exec(GeoNodeExecParams params)
{
  GeometrySet geometry = params.extract_input<GeometrySet>("Geometry");
  geometry = geometry_set_realize_instances(geometry);
  Field<bool> selection_field = params.extract_input<Field<bool>>("Selection");
  Field<float3> positions_field = params.extract_input<Field<float3>>("Position");

  try_set_position_in_component(
      geometry, GEO_COMPONENT_TYPE_MESH, selection_field, positions_field);

  params.set_output("Geometry", std::move(geometry));
}

}  // namespace blender::nodes

void register_node_type_geo_set_position()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_SET_POSITION, "Set Position", NODE_CLASS_GEOMETRY, 0);
  ntype.geometry_node_execute = blender::nodes::geo_node_set_position_exec;
  ntype.declare = blender::nodes::geo_node_set_position_declare;
  nodeRegisterType(&ntype);
}
