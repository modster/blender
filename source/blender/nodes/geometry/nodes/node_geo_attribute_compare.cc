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

#include "node_geometry_util.hh"

#include "BKE_attribute.h"
#include "BKE_attribute_access.hh"

#include "BLI_array.hh"
#include "BLI_math_base_safe.h"
#include "BLI_rand.hh"

#include "DNA_mesh_types.h"
#include "DNA_pointcloud_types.h"

#include "NOD_math_functions.hh"

static bNodeSocketTemplate geo_node_attribute_compare_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_STRING, N_("Attribute A")},
    {SOCK_FLOAT, N_("A"), 0.0f, 0.0f, 0.0f, 0.0f, -FLT_MAX, FLT_MAX},
    {SOCK_STRING, N_("Attribute B")},
    {SOCK_FLOAT, N_("B"), 0.0f, 0.0f, 0.0f, 0.0f, -FLT_MAX, FLT_MAX},
    {SOCK_FLOAT, N_("Epsilon"), 0.01f, 0.0f, 0.0f, 0.0f, 0.0f, FLT_MAX},
    {SOCK_STRING, N_("Result")},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_attribute_compare_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

static void geo_node_attribute_compare_init(bNodeTree *UNUSED(tree), bNode *node)
{
  node->custom1 = NODE_FLOAT_COMPARE_GREATER_THAN;
  node->custom2 = GEO_NODE_USE_ATTRIBUTE_A | GEO_NODE_USE_ATTRIBUTE_B;
}

static bool use_epsilon(const bNode &node)
{
  return ELEM(node.custom1, NODE_FLOAT_COMPARE_EQUAL, NODE_FLOAT_COMPARE_NOT_EQUAL);
}

static void geo_node_attribute_compare_update(bNodeTree *UNUSED(ntree), bNode *node)
{
  bNodeSocket *socket_attribute_a = (bNodeSocket *)BLI_findlink(&node->inputs, 1);
  bNodeSocket *socket_float_a = socket_attribute_a->next;
  bNodeSocket *socket_attribute_b = socket_float_a->next;
  bNodeSocket *socket_float_b = socket_attribute_b->next;
  bNodeSocket *socket_epsilon = socket_float_b->next;

  GeometryNodeUseAttributeFlag flag = static_cast<GeometryNodeUseAttributeFlag>(node->custom2);

  nodeSetSocketAvailability(socket_attribute_a, flag & GEO_NODE_USE_ATTRIBUTE_A);
  nodeSetSocketAvailability(socket_attribute_b, flag & GEO_NODE_USE_ATTRIBUTE_B);
  nodeSetSocketAvailability(socket_float_a, !(flag & GEO_NODE_USE_ATTRIBUTE_A));
  nodeSetSocketAvailability(socket_float_b, !(flag & GEO_NODE_USE_ATTRIBUTE_B));
  nodeSetSocketAvailability(socket_epsilon, use_epsilon(*node));
}

namespace blender::nodes {

static void do_math_operation(const FloatReadAttribute &input_a,
                              const FloatReadAttribute &input_b,
                              BooleanWriteAttribute result,
                              const int operation,
                              const float epsilon)
{
  const int size = input_a.size();

  Span<float> span_a = input_a.get_span();
  Span<float> span_b = input_b.get_span();
  MutableSpan<bool> span_result = result.get_span();

  if (try_dispatch_float_math_fl_fl_to_bool(
          operation, [&](auto math_function, const FloatMathOperationInfo &UNUSED(info)) {
            for (const int i : IndexRange(size)) {
              const float in1 = span_a[i];
              const float in2 = span_b[i];
              const bool out = math_function(in1, in2);
              span_result[i] = out;
            }
          })) {
    result.apply_span();
    return;
  }

  if (try_dispatch_float_math_fl_fl_fl_to_bool(
          operation, [&](auto math_function, const FloatMathOperationInfo &UNUSED(info)) {
            for (const int i : IndexRange(size)) {
              const float in1 = span_a[i];
              const float in2 = span_b[i];
              const bool out = math_function(in1, in2, epsilon);
              span_result[i] = out;
            }
          })) {
    result.apply_span();
    return;
  }

  /* The operation is not supported by this node currently. */
  BLI_assert(false);
}

static void attribute_compare_calc(GeometryComponent &component, const GeoNodeExecParams &params)
{
  const bNode &node = params.node();
  const int operation = node.custom1;

  /* The result type of this node is always float. */
  const CustomDataType result_type = CD_PROP_BOOL;
  /* The result domain is always point for now. */
  const AttributeDomain result_domain = ATTR_DOMAIN_POINT;

  /* Get result attribute first, in case it has to overwrite one of the existing attributes. */
  const std::string result_name = params.get_input<std::string>("Result");
  WriteAttributePtr attribute_result = component.attribute_try_ensure_for_write(
      result_name, result_domain, result_type);
  if (!attribute_result) {
    return;
  }

  GeometryNodeUseAttributeFlag flag = static_cast<GeometryNodeUseAttributeFlag>(node.custom2);

  auto get_input_attribute = [&](GeometryNodeUseAttributeFlag use_flag,
                                 StringRef attribute_socket_identifier,
                                 StringRef value_socket_identifier) {
    if (flag & use_flag) {
      const std::string attribute_name = params.get_input<std::string>(
          attribute_socket_identifier);
      return component.attribute_try_get_for_read(attribute_name, result_domain, CD_PROP_FLOAT);
    }
    const float value = params.get_input<float>(value_socket_identifier);
    return component.attribute_get_constant_for_read(result_domain, CD_PROP_FLOAT, &value);
  };

  ReadAttributePtr attribute_a = get_input_attribute(GEO_NODE_USE_ATTRIBUTE_A, "Attribute A", "A");
  ReadAttributePtr attribute_b = get_input_attribute(GEO_NODE_USE_ATTRIBUTE_B, "Attribute B", "B");

  if (!attribute_a || !attribute_b) {
    /* Attribute wasn't found. */
    return;
  }

  const float epsilon = use_epsilon(node) ? params.get_input<float>("Epsilon") : 0.0f;

  do_math_operation(std::move(attribute_a),
                    std::move(attribute_b),
                    std::move(attribute_result),
                    operation,
                    epsilon);
}

static void geo_node_attribute_compare_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");

  if (geometry_set.has<MeshComponent>()) {
    attribute_compare_calc(geometry_set.get_component_for_write<MeshComponent>(), params);
  }
  if (geometry_set.has<PointCloudComponent>()) {
    attribute_compare_calc(geometry_set.get_component_for_write<PointCloudComponent>(), params);
  }

  params.set_output("Geometry", geometry_set);
}

}  // namespace blender::nodes

void register_node_type_geo_attribute_compare()
{
  static bNodeType ntype;

  geo_node_type_base(
      &ntype, GEO_NODE_ATTRIBUTE_COMPARE, "Attribute Compare", NODE_CLASS_ATTRIBUTE, 0);
  node_type_socket_templates(
      &ntype, geo_node_attribute_compare_in, geo_node_attribute_compare_out);
  ntype.geometry_node_execute = blender::nodes::geo_node_attribute_compare_exec;
  node_type_update(&ntype, geo_node_attribute_compare_update);
  node_type_init(&ntype, geo_node_attribute_compare_init);
  nodeRegisterType(&ntype);
}
