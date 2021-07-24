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
#include "BLI_kdtree.h"
#include "BLI_span.hh"

#include "DNA_pointcloud_types.h"

#include "GEO_mesh_merge_by_distance.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "node_geometry_util.hh"

#include "FN_generic_span.hh"

#include "node_geometry_util.hh"

using blender::float3;
using blender::Span;
using blender::Vector;

static bNodeSocketTemplate geo_node_merge_by_distance_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_FLOAT, N_("Distance"), 0.0f, 0, 0, 0, 0, 10000.0f, PROP_DISTANCE},
    {SOCK_STRING, N_("Selection")},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_merge_by_distance_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

static void geo_node_merge_by_distance_layout(uiLayout *layout,
                                              bContext *UNUSED(C),
                                              PointerRNA *ptr)
{
  uiItemR(layout, ptr, "merge_mode", 0, "", ICON_NONE);
}

static void geo_merge_by_distance_init(bNodeTree *UNUSED(ntree), bNode *node)
{
  node->custom1 = WELD_MODE_ALL;
}

static KDTree_3d *build_kdtree(Span<float3> positions)
{
  KDTree_3d *kdtree = BLI_kdtree_3d_new(positions.size());

  int i_point = 0;
  for (const float3 position : positions) {
    BLI_kdtree_3d_insert(kdtree, i_point, position);
    i_point++;
  }
  BLI_kdtree_3d_balance(kdtree);
  return kdtree;
}

static void build_merge_map(Vector<float3> &positions,
                            Vector<int> &merge_map,
                            int &total_merge_operations,
                            const float merge_threshold,
                            Span<bool> &selection)
{
  KDTree_3d *kdtree = build_kdtree(positions.as_span());

  for (int i : positions.index_range()) {
    struct CallbackData {
      int index;
      int &total_merge_operations;
      Vector<int> &merge_map;
      Span<bool> &selection;
    } callback_data = {i, total_merge_operations, merge_map, selection};

    BLI_kdtree_3d_range_search_cb(
        kdtree,
        positions[i],
        merge_threshold,
        [](void *user_data,
           int source_vertex_index,
           const float *UNUSED(co),
           float UNUSED(distance_squared)) {
          CallbackData &callback_data = *static_cast<CallbackData *>(user_data);
          int target_vertex_index = callback_data.index;
          if (source_vertex_index != target_vertex_index &&
              callback_data.merge_map[source_vertex_index] == -1 &&
              callback_data.merge_map[target_vertex_index] == -1 &&
              callback_data.selection[source_vertex_index] &&
              callback_data.selection[target_vertex_index]) {
            callback_data.merge_map[source_vertex_index] = target_vertex_index;
            callback_data.total_merge_operations++;
          }
          return true;
        },
        &callback_data);
  }

  BLI_kdtree_3d_free(kdtree);
}

static PointCloud *merge_by_distance_pointcloud(const PointCloud *point_cloud,
                                                const float merge_threshold,
                                                Vector<int> &merge_map,
                                                Span<bool> &selection)
{
  Vector<float3> positions;
  for (int i = 0; i < point_cloud->totpoint; i++) {
    positions.append(point_cloud->co[i]);
  }

  merge_map = Vector<int>(point_cloud->totpoint, -1);
  int total_merge_operations = 0;

  BLI_assert(positions.size() == merge_map.size());

  build_merge_map(positions, merge_map, total_merge_operations, merge_threshold, selection);

  PointCloud *result = BKE_pointcloud_new_nomain(positions.size() - total_merge_operations);
  int offset = 0;
  for (const int i : positions.index_range()) {
    if (merge_map[i] == -1) {
      copy_v3_v3(result->co[offset], positions[i]);
      result->radius[offset] = point_cloud->radius[i];
      offset++;
    }
  }

  return result;
}

namespace blender::nodes {
static void geo_node_merge_by_distance_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");
  geometry_set = geometry_set_realize_instances(geometry_set);

  const char weld_mode = params.node().custom1;
  const float distance = params.extract_input<float>("Distance");

  if (geometry_set.has_mesh()) {
    MeshComponent &mesh_component = geometry_set.get_component_for_write<MeshComponent>();
    const Mesh *input_mesh = mesh_component.get_for_read();

    const bool default_selection = true;
    GVArray_Typed<bool> selection_attribute = params.get_input_attribute<bool>(
        "Selection", mesh_component, ATTR_DOMAIN_POINT, default_selection);
    VArray_Span<bool> selection{selection_attribute};

    Mesh *result = GEO_mesh_merge_by_distance(input_mesh, selection.data(), distance, weld_mode);
    if (result != input_mesh) {
      geometry_set.replace_mesh(result);
    }
  }

  if (geometry_set.has_pointcloud()) {
    PointCloudComponent &point_cloud_component =
        geometry_set.get_component_for_write<PointCloudComponent>();
    const PointCloud *point_cloud = point_cloud_component.get_for_write();
    const bool default_selection = true;
    GVArray_Typed<bool> selection_attribute = params.get_input_attribute<bool>(
        "Selection", point_cloud_component, ATTR_DOMAIN_POINT, default_selection);
    VArray_Span<bool> selection{selection_attribute};
    Vector<int> merge_map;
    point_cloud_component.replace(merge_by_distance_pointcloud(point_cloud, distance, merge_map, selection));
  }

  if (geometry_set.has_volume()) {
    params.error_message_add(NodeWarningType::Warning,
                             TIP_("This Node does not operate on volumes"));
  }

  params.set_output("Geometry", std::move(geometry_set));
}
}  // namespace blender::nodes

void register_node_type_geo_merge_by_distance()
{
  static bNodeType ntype;
  geo_node_type_base(
      &ntype, GEO_NODE_MERGE_BY_DISTANCE, "Merge By Distance", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(
      &ntype, geo_node_merge_by_distance_in, geo_node_merge_by_distance_out);
  node_type_init(&ntype, geo_merge_by_distance_init);
  ntype.geometry_node_execute = blender::nodes::geo_node_merge_by_distance_exec;
  ntype.draw_buttons = geo_node_merge_by_distance_layout;
  nodeRegisterType(&ntype);
}
