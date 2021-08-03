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

#include "BLI_kdopbvh.h"
#include "BLI_kdtree.h"
#include "BLI_task.hh"
#include "BLI_timeit.hh"

#include "DNA_mesh_types.h"

#include "BKE_bvhutils.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "node_geometry_util.hh"

static bNodeSocketTemplate geo_node_attribute_proximity_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_GEOMETRY, N_("Target")},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_attribute_proximity_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_FLOAT, N_("Distance")},
    {SOCK_VECTOR, N_("Position")},
    {-1, ""},
};

static void geo_node_attribute_proximity_layout(uiLayout *layout,
                                                bContext *UNUSED(C),
                                                PointerRNA *ptr)
{
  uiItemR(layout, ptr, "target_geometry_element", 0, "", ICON_NONE);
}

static void geo_attribute_proximity_init(bNodeTree *UNUSED(ntree), bNode *node)
{
  NodeGeometryAttributeProximity *node_storage = (NodeGeometryAttributeProximity *)MEM_callocN(
      sizeof(NodeGeometryAttributeProximity), __func__);

  node_storage->target_geometry_element =
      GEO_NODE_ATTRIBUTE_PROXIMITY_TARGET_GEOMETRY_ELEMENT_FACES;
  node->storage = node_storage;
}

namespace blender::nodes {

static void proximity_calc(MutableSpan<float> distance_span,
                           MutableSpan<float3> location_span,
                           const VArray<float3> &positions,
                           BVHTreeFromMesh &tree_data_mesh,
                           BVHTreeFromPointCloud &tree_data_pointcloud,
                           const bool bvh_mesh_success,
                           const bool bvh_pointcloud_success)
{
  IndexRange range = positions.index_range();
  threading::parallel_for(range, 512, [&](IndexRange range) {
    BVHTreeNearest nearest_from_mesh;
    BVHTreeNearest nearest_from_pointcloud;

    copy_v3_fl(nearest_from_mesh.co, FLT_MAX);
    copy_v3_fl(nearest_from_pointcloud.co, FLT_MAX);

    nearest_from_mesh.index = -1;
    nearest_from_pointcloud.index = -1;

    for (int i : range) {
      /* Use the distance to the last found point as upper bound to speedup the bvh lookup. */
      nearest_from_mesh.dist_sq = len_squared_v3v3(nearest_from_mesh.co, positions[i]);

      if (bvh_mesh_success) {
        BLI_bvhtree_find_nearest(tree_data_mesh.tree,
                                 positions[i],
                                 &nearest_from_mesh,
                                 tree_data_mesh.nearest_callback,
                                 &tree_data_mesh);
      }

      /* Use the distance to the closest point in the mesh to speedup the pointcloud bvh lookup.
       * This is ok because we only need to find the closest point in the pointcloud if it's closer
       * than the mesh. */
      nearest_from_pointcloud.dist_sq = nearest_from_mesh.dist_sq;

      if (bvh_pointcloud_success) {
        BLI_bvhtree_find_nearest(tree_data_pointcloud.tree,
                                 positions[i],
                                 &nearest_from_pointcloud,
                                 tree_data_pointcloud.nearest_callback,
                                 &tree_data_pointcloud);
      }

      if (nearest_from_pointcloud.dist_sq < nearest_from_mesh.dist_sq) {
        if (!distance_span.is_empty()) {
          distance_span[i] = sqrtf(nearest_from_pointcloud.dist_sq);
        }
        if (!location_span.is_empty()) {
          location_span[i] = nearest_from_pointcloud.co;
        }
      }
      else {
        if (!distance_span.is_empty()) {
          distance_span[i] = sqrtf(nearest_from_mesh.dist_sq);
        }
        if (!location_span.is_empty()) {
          location_span[i] = nearest_from_mesh.co;
        }
      }
    }
  });
}

static bool bvh_from_mesh(const Mesh *target_mesh,
                          int target_geometry_element,
                          BVHTreeFromMesh &r_tree_data_mesh)
{
  BVHCacheType bvh_type = BVHTREE_FROM_LOOPTRI;
  switch (target_geometry_element) {
    case GEO_NODE_ATTRIBUTE_PROXIMITY_TARGET_GEOMETRY_ELEMENT_POINTS:
      bvh_type = BVHTREE_FROM_VERTS;
      break;
    case GEO_NODE_ATTRIBUTE_PROXIMITY_TARGET_GEOMETRY_ELEMENT_EDGES:
      bvh_type = BVHTREE_FROM_EDGES;
      break;
    case GEO_NODE_ATTRIBUTE_PROXIMITY_TARGET_GEOMETRY_ELEMENT_FACES:
      bvh_type = BVHTREE_FROM_LOOPTRI;
      break;
  }

  BKE_bvhtree_from_mesh_get(&r_tree_data_mesh, target_mesh, bvh_type, 2);
  if (r_tree_data_mesh.tree == nullptr) {
    return false;
  }
  return true;
}

static bool bvh_from_pointcloud(const PointCloud *target_pointcloud,
                                BVHTreeFromPointCloud &r_tree_data_pointcloud)
{
  BKE_bvhtree_from_pointcloud_get(&r_tree_data_pointcloud, target_pointcloud, 2);
  if (r_tree_data_pointcloud.tree == nullptr) {
    return false;
  }
  return true;
}

static void attribute_calc_proximity(GeometryComponent &component,
                                     const GeometrySet &geometry_set_target,
                                     const AnonymousCustomDataLayerID *distance_id,
                                     const AnonymousCustomDataLayerID *location_id,
                                     const GeoNodeExecParams &params)
{
  GVArray_Typed<float3> positions = component.attribute_get_for_read<float3>(
      "position", ATTR_DOMAIN_POINT, {0, 0, 0});

  std::optional<OutputAttribute_Typed<float3>> location_attribute;
  std::optional<OutputAttribute_Typed<float>> distance_attribute;
  if (location_id != nullptr) {
    location_attribute.emplace(component.attribute_try_get_anonymous_for_output_only<float3>(
        *location_id, ATTR_DOMAIN_POINT));
  }
  if (distance_id != nullptr) {
    distance_attribute.emplace(component.attribute_try_get_anonymous_for_output_only<float>(
        *distance_id, ATTR_DOMAIN_POINT));
  }

  const bNode &node = params.node();
  const NodeGeometryAttributeProximity &storage = *(const NodeGeometryAttributeProximity *)
                                                       node.storage;

  BVHTreeFromMesh tree_data_mesh;
  BVHTreeFromPointCloud tree_data_pointcloud;
  bool bvh_mesh_success = false;
  bool bvh_pointcloud_success = false;

  if (geometry_set_target.has_mesh()) {
    bvh_mesh_success = bvh_from_mesh(
        geometry_set_target.get_mesh_for_read(), storage.target_geometry_element, tree_data_mesh);
  }

  if (geometry_set_target.has_pointcloud() &&
      storage.target_geometry_element ==
          GEO_NODE_ATTRIBUTE_PROXIMITY_TARGET_GEOMETRY_ELEMENT_POINTS) {
    bvh_pointcloud_success = bvh_from_pointcloud(geometry_set_target.get_pointcloud_for_read(),
                                                 tree_data_pointcloud);
  }

  proximity_calc(distance_attribute ? distance_attribute->as_span() : MutableSpan<float>(),
                 location_attribute ? location_attribute->as_span() : MutableSpan<float3>(),
                 positions,
                 tree_data_mesh,
                 tree_data_pointcloud,
                 bvh_mesh_success,
                 bvh_pointcloud_success);

  if (bvh_mesh_success) {
    free_bvhtree_from_mesh(&tree_data_mesh);
  }
  if (bvh_pointcloud_success) {
    free_bvhtree_from_pointcloud(&tree_data_pointcloud);
  }

  if (location_attribute) {
    location_attribute->save();
  }
  if (distance_attribute) {
    for (const int i : IndexRange(distance_attribute->as_span().size())) {
      std::cout << distance_attribute->as_span()[i] << "\n";
    }
    distance_attribute->save();
  }
}

static void geo_node_attribute_proximity_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");
  GeometrySet geometry_set_target = params.extract_input<GeometrySet>("Target");

  geometry_set = geometry_set_realize_instances(geometry_set);

  /* This isn't required. This node should be rewritten to handle instances
   * for the target geometry set. However, the generic BVH API complicates this. */
  geometry_set_target = geometry_set_realize_instances(geometry_set_target);

  AnonymousCustomDataLayerID *distance = params.output_is_required("Distance") ?
                                             CustomData_anonymous_id_new("Distance") :
                                             nullptr;
  AnonymousCustomDataLayerID *location = params.output_is_required("Position") ?
                                             CustomData_anonymous_id_new("Position") :
                                             nullptr;
  if (!distance && !location) {
    params.set_output("Geometry", std::move(geometry_set));
    return;
  }

  static const Array<GeometryComponentType> types = {
      GEO_COMPONENT_TYPE_MESH, GEO_COMPONENT_TYPE_POINT_CLOUD, GEO_COMPONENT_TYPE_CURVE};
  for (const GeometryComponentType type : types) {
    if (geometry_set.has(type)) {
      attribute_calc_proximity(geometry_set.get_component_for_write(type),
                               geometry_set_target,
                               distance,
                               location,
                               params);
    }
  }

  if (distance != nullptr) {
    params.set_output(
        "Distance",
        bke::FieldRef<float>(new bke::AnonymousAttributeField(*distance, CPPType::get<float>())));
  }
  if (location != nullptr) {
    params.set_output("Position",
                      bke::FieldRef<float3>(
                          new bke::AnonymousAttributeField(*location, CPPType::get<float3>())));
  }

  params.set_output("Geometry", std::move(geometry_set));
}

}  // namespace blender::nodes

void register_node_type_geo_attribute_proximity()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_ATTRIBUTE_PROXIMITY, "Proximity", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(
      &ntype, geo_node_attribute_proximity_in, geo_node_attribute_proximity_out);
  node_type_init(&ntype, geo_attribute_proximity_init);
  node_type_storage(&ntype,
                    "NodeGeometryAttributeProximity",
                    node_free_standard_storage,
                    node_copy_standard_storage);
  ntype.geometry_node_execute = blender::nodes::geo_node_attribute_proximity_exec;
  ntype.draw_buttons = geo_node_attribute_proximity_layout;
  nodeRegisterType(&ntype);
}
