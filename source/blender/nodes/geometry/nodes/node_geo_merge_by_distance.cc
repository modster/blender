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

#include "BKE_pointcloud.h"

#include "BLI_float3.hh"
#include "BLI_span.hh"
#include "BLI_task.hh"

#include "DNA_mesh_types.h"
#include "DNA_pointcloud_types.h"

#include "GEO_mesh_merge_by_distance.hh"
#include "GEO_pointcloud_merge_by_distance.hh"

#include "UI_interface.h"
#include "UI_resources.h"

#include "node_geometry_util.hh"

using blender::Array;
using blender::float3;
using blender::Span;
using blender::Vector;

namespace blender::nodes {
static void geo_node_merge_by_distance_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Geometry");
  b.add_input<decl::Float>("Distance").min(0.0f).max(10000.0f);
  b.add_input<decl::Bool>("Selection").default_value(true).hide_value().supports_field();

  b.add_output<decl::Geometry>("Geometry");
}

static void geo_node_merge_by_distance_layout(uiLayout *layout,
                                              bContext *UNUSED(C),
                                              PointerRNA *ptr)
{
  uiItemR(layout, ptr, "merge_mode", 0, "", ICON_NONE);
}

static void geo_merge_by_distance_init(bNodeTree *UNUSED(ntree), bNode *node)
{
  node->custom1 = weld_mode_to_int(geometry::WeldMode::all);
}

static void process_mesh(GeoNodeExecParams &params,
                         const geometry::WeldMode weld_mode,
                         const float distance,
                         GeometrySet &geometry_set)
{
  MeshComponent &mesh_component = geometry_set.get_component_for_write<MeshComponent>();
  Mesh *input_mesh = mesh_component.get_for_write();

  GeometryComponentFieldContext field_context{mesh_component, ATTR_DOMAIN_POINT};
  const Field<bool> selection_field = params.extract_input<Field<bool>>("Selection");
  fn::FieldEvaluator selection_evaluator{field_context, input_mesh->totvert};
  selection_evaluator.add(selection_field);
  selection_evaluator.evaluate();
  const VArray_Span<bool> selection = selection_evaluator.get_evaluated<bool>(0);

  Mesh *result = geometry::mesh_merge_by_distance(input_mesh, selection, distance, weld_mode);
  if (result != input_mesh) {
    geometry_set.replace_mesh(result);
  }
}

static void process_pointcloud(GeoNodeExecParams &params,
                               const float distance,
                               GeometrySet &geometry_set)
{
  PointCloudComponent &pointcloud_component =
      geometry_set.get_component_for_write<PointCloudComponent>();
  const PointCloud &pointcloud = *pointcloud_component.get_for_read();

  GeometryComponentFieldContext field_context{pointcloud_component, ATTR_DOMAIN_POINT};
  const Field<bool> selection_field = params.extract_input<Field<bool>>("Selection");
  fn::FieldEvaluator selection_evaluator{field_context, pointcloud.totpoint};
  selection_evaluator.add(selection_field);
  selection_evaluator.evaluate();
  const VArray_Span<bool> selection = selection_evaluator.get_evaluated<bool>(0);

  pointcloud_component.replace(
      geometry::pointcloud_merge_by_distance(pointcloud_component, distance, selection));
}

static void geo_node_merge_by_distance_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");

  const geometry::WeldMode weld_mode = geometry::weld_mode_from_int(params.node().custom1);
  const float distance = params.extract_input<float>("Distance");

  if (geometry_set.has_instances()) {
    InstancesComponent &instances = geometry_set.get_component_for_write<InstancesComponent>();
    instances.ensure_geometry_instances();

    threading::parallel_for(IndexRange(instances.references_amount()), 16, [&](IndexRange range) {
      for (int i : range) {
        GeometrySet &geometry_set = instances.geometry_set_from_reference(i);
        geometry_set = bke::geometry_set_realize_instances(geometry_set);

        if (geometry_set.has_mesh()) {
          process_mesh(params, weld_mode, distance, geometry_set);
        }

        if (geometry_set.has_pointcloud()) {
          process_pointcloud(params, distance, geometry_set);
        }
      }
    });
  }
  else {
    if (geometry_set.has_mesh()) {
      process_mesh(params, weld_mode, distance, geometry_set);
    }

    if (geometry_set.has_pointcloud()) {
      process_pointcloud(params, distance, geometry_set);
    }
  }

  params.set_output("Geometry", std::move(geometry_set));
}
}  // namespace blender::nodes

void register_node_type_geo_merge_by_distance()
{
  static bNodeType ntype;
  geo_node_type_base(
      &ntype, GEO_NODE_MERGE_BY_DISTANCE, "Merge By Distance", NODE_CLASS_GEOMETRY, 0);
  node_type_init(&ntype, blender::nodes::geo_merge_by_distance_init);
  ntype.declare = blender::nodes::geo_node_merge_by_distance_declare;
  ntype.geometry_node_execute = blender::nodes::geo_node_merge_by_distance_exec;
  ntype.draw_buttons = blender::nodes::geo_node_merge_by_distance_layout;
  nodeRegisterType(&ntype);
}
