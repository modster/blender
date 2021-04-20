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

#include "BLI_kdopbvh.h"

#include "BKE_bvhutils.h"

#include "DNA_mesh_types.h"
#include "DNA_pointcloud_types.h"

#include "node_geometry_util.hh"

static bNodeSocketTemplate geo_node_attribute_transfer_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_GEOMETRY, N_("Target")},
    {SOCK_STRING, N_("Source")},
    {SOCK_STRING, N_("Destination")},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_attribute_transfer_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

static void geo_node_attribute_transfer_layout(uiLayout *layout,
                                               bContext *UNUSED(C),
                                               PointerRNA *ptr)
{
  uiLayoutSetPropSep(layout, true);
  uiLayoutSetPropDecorate(layout, false);
  uiItemR(layout, ptr, "domain", 0, IFACE_("Domain"), ICON_NONE);
  uiItemR(layout, ptr, "mapping", 0, IFACE_("Mapping"), ICON_NONE);
}

namespace blender::nodes {

static void geo_node_attribute_transfer_init(bNodeTree *UNUSED(tree), bNode *node)
{
  NodeGeometryAttributeTransfer *data = (NodeGeometryAttributeTransfer *)MEM_callocN(
      sizeof(NodeGeometryAttributeTransfer), __func__);
  node->storage = data;
}

static CustomDataType get_result_data_type(const GeometrySet &geometry,
                                           const StringRef attribute_name)
{
  Vector<CustomDataType> data_types;

  const PointCloudComponent *pointcloud_component =
      geometry.get_component_for_read<PointCloudComponent>();
  if (pointcloud_component != nullptr) {
    ReadAttributeLookup attribute = pointcloud_component->attribute_try_get_for_read(
        attribute_name);
    if (attribute) {
      data_types.append(bke::cpp_type_to_custom_data_type(attribute.varray->type()));
    }
  }

  const MeshComponent *mesh_component = geometry.get_component_for_read<MeshComponent>();
  if (mesh_component != nullptr) {
    ReadAttributeLookup attribute = mesh_component->attribute_try_get_for_read(attribute_name);
    if (attribute) {
      data_types.append(bke::cpp_type_to_custom_data_type(attribute.varray->type()));
    }
  }
  return bke::attribute_data_type_highest_complexity(data_types);
}

static void get_closest_pointcloud_point_indices(const PointCloud &pointcloud,
                                                 const VArray<float3> &positions,
                                                 const MutableSpan<int> r_indices)
{
  BLI_assert(positions.size() == r_indices.size());
  BLI_assert(pointcloud.totpoint > 0);

  BVHTreeFromPointCloud tree_data;
  BKE_bvhtree_from_pointcloud_get(&tree_data, &pointcloud, 2);

  for (const int i : positions.index_range()) {
    BVHTreeNearest nearest;
    nearest.dist_sq = FLT_MAX;
    const float3 position = positions[i];
    BLI_bvhtree_find_nearest(
        tree_data.tree, position, &nearest, tree_data.nearest_callback, &tree_data);
    r_indices[i] = nearest.index;
  }

  free_bvhtree_from_pointcloud(&tree_data);
}

static void get_closest_mesh_point_indices(const Mesh &mesh,
                                           const VArray<float3> &positions,
                                           const MutableSpan<int> r_indices)
{
  BLI_assert(positions.size() == r_indices.size());
  BLI_assert(mesh.totvert > 0);

  BVHTreeFromMesh tree_data;
  BKE_bvhtree_from_mesh_get(&tree_data, const_cast<Mesh *>(&mesh), BVHTREE_FROM_VERTS, 2);

  for (const int i : positions.index_range()) {
    BVHTreeNearest nearest;
    nearest.dist_sq = FLT_MAX;
    const float3 position = positions[i];
    BLI_bvhtree_find_nearest(
        tree_data.tree, position, &nearest, tree_data.nearest_callback, &tree_data);
    r_indices[i] = nearest.index;
  }

  free_bvhtree_from_mesh(&tree_data);
}

static void get_closest_mesh_surface_samples(const Mesh &mesh,
                                             const VArray<float3> &positions,
                                             const MutableSpan<int> r_looptri_indices,
                                             const MutableSpan<float3> r_positions)
{
  BLI_assert(positions.size() == r_looptri_indices.size());
  BLI_assert(positions.size() == r_positions.size());

  BVHTreeFromMesh tree_data;
  BKE_bvhtree_from_mesh_get(&tree_data, const_cast<Mesh *>(&mesh), BVHTREE_FROM_LOOPTRI, 2);

  for (const int i : positions.index_range()) {
    BVHTreeNearest nearest;
    nearest.dist_sq = FLT_MAX;
    const float3 position = positions[i];
    BLI_bvhtree_find_nearest(
        tree_data.tree, position, &nearest, tree_data.nearest_callback, &tree_data);
    r_looptri_indices[i] = nearest.index;
    r_positions[i] = nearest.co;
  }

  free_bvhtree_from_mesh(&tree_data);
}

static void transfer_attribute(const GeometrySet &src_geometry,
                               GeometryComponent &dst_component,
                               const AttributeDomain result_domain,
                               const CustomDataType data_type,
                               const StringRef src_name,
                               const StringRef dst_name)
{
  const CPPType &type = *bke::custom_data_type_to_cpp_type(data_type);

  if (!src_geometry.has<PointCloudComponent>()) {
    return;
  }
  const PointCloudComponent &src_component =
      *src_geometry.get_component_for_read<PointCloudComponent>();
  const PointCloud *pointcloud = src_component.get_for_read();
  if (pointcloud == nullptr) {
    return;
  }
  if (pointcloud->totpoint == 0) {
    return;
  }
  ReadAttributeLookup src_attribute = src_component.attribute_try_get_for_read(src_name);
  if (!src_attribute) {
    return;
  }
  /* TODO: Possibly convert data type. */
  BLI_assert(src_attribute.varray->type() == type);

  OutputAttribute dst_attribute = dst_component.attribute_try_get_for_output_only(
      dst_name, result_domain, data_type);
  if (!dst_attribute) {
    return;
  }

  GVArray_Typed<float3> dst_positions = dst_component.attribute_get_for_read<float3>(
      "position", result_domain, {0, 0, 0});

  Array<int> nearest_point_indices(dst_positions.size());
  get_closest_pointcloud_point_indices(*pointcloud, dst_positions, nearest_point_indices);

  BUFFER_FOR_CPP_TYPE_VALUE(type, buffer);

  for (const int i : dst_positions.index_range()) {
    const int point_index = nearest_point_indices[i];
    src_attribute.varray->get(point_index, buffer);
    dst_attribute->set_by_relocate(i, buffer);
  }

  dst_attribute.save();
}

static void geo_node_attribute_transfer_exec(GeoNodeExecParams params)
{
  GeometrySet dst_geometry_set = params.extract_input<GeometrySet>("Geometry");
  GeometrySet src_geometry_set = params.extract_input<GeometrySet>("Target");
  const std::string src_attribute_name = params.extract_input<std::string>("Source");
  const std::string dst_attribute_name = params.extract_input<std::string>("Destination");

  if (src_attribute_name.empty() || dst_attribute_name.empty()) {
    params.set_output("Geometry", dst_geometry_set);
    return;
  }

  const NodeGeometryAttributeTransfer &storage =
      *(const NodeGeometryAttributeTransfer *)params.node().storage;
  const AttributeDomain dst_domain = (AttributeDomain)storage.domain;
  const GeometryNodeAttributeTransferMappingMode mapping =
      (GeometryNodeAttributeTransferMappingMode)storage.mapping;

  dst_geometry_set = bke::geometry_set_realize_instances(dst_geometry_set);
  src_geometry_set = bke::geometry_set_realize_instances(src_geometry_set);

  const CustomDataType result_data_type = get_result_data_type(src_geometry_set,
                                                               src_attribute_name);

  if (dst_geometry_set.has<MeshComponent>()) {
    transfer_attribute(src_geometry_set,
                       dst_geometry_set.get_component_for_write<MeshComponent>(),
                       dst_domain,
                       result_data_type,
                       src_attribute_name,
                       dst_attribute_name);
  }
  if (dst_geometry_set.has<PointCloudComponent>()) {
    transfer_attribute(src_geometry_set,
                       dst_geometry_set.get_component_for_write<PointCloudComponent>(),
                       dst_domain,
                       result_data_type,
                       src_attribute_name,
                       dst_attribute_name);
  }

  params.set_output("Geometry", dst_geometry_set);
}

}  // namespace blender::nodes

void register_node_type_geo_attribute_transfer()
{
  static bNodeType ntype;

  geo_node_type_base(
      &ntype, GEO_NODE_ATTRIBUTE_TRANSFER, "Attribute Transfer", NODE_CLASS_ATTRIBUTE, 0);
  node_type_socket_templates(
      &ntype, geo_node_attribute_transfer_in, geo_node_attribute_transfer_out);
  node_type_init(&ntype, blender::nodes::geo_node_attribute_transfer_init);
  node_type_storage(&ntype,
                    "NodeGeometryAttributeTransfer",
                    node_free_standard_storage,
                    node_copy_standard_storage);
  ntype.geometry_node_execute = blender::nodes::geo_node_attribute_transfer_exec;
  ntype.draw_buttons = geo_node_attribute_transfer_layout;
  nodeRegisterType(&ntype);
}
