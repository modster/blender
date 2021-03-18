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

#include "BKE_attribute_math.hh"
#include "BKE_mesh.h"
#include "BKE_persistent_data_handle.hh"
#include "BKE_pointcloud.h"

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"
#include "DNA_pointcloud_types.h"

#include "node_geometry_util.hh"

static bNodeSocketTemplate geo_node_point_instance_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_STRING, N_("Mask")},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_point_instance_out[] = {
    {SOCK_GEOMETRY, N_("Geometry 1")},
    {SOCK_GEOMETRY, N_("Geometry 2")},
    {-1, ""},
};

using blender::bke::AttributeKind;
using blender::bke::GeometryInstanceGroup;

namespace blender::nodes {

static void gather_positions_from_component_instances(const GeometryComponent &component,
                                                      const StringRef mask_attribute_name,
                                                      Span<float4x4> transforms,
                                                      MutableSpan<Vector<float3>> r_positions_a,
                                                      MutableSpan<Vector<float3>> r_positions_b)
{
  if (component.attribute_domain_size(ATTR_DOMAIN_POINT) == 0) {
    return;
  }

  const BooleanReadAttribute mask_attribute = component.attribute_get_for_read<bool>(
      mask_attribute_name, ATTR_DOMAIN_POINT, false);
  const Float3ReadAttribute position_attribute = component.attribute_get_for_read<float3>(
      "position", ATTR_DOMAIN_POINT, {0.0f, 0.0f, 0.0f});

  Span<bool> masks = mask_attribute.get_span();
  Span<float3> component_positions = position_attribute.get_span();

  for (const int instance_index : transforms.index_range()) {
    const float4x4 &transform = transforms[instance_index];
    for (const int i : masks.index_range()) {
      if (masks[i]) {
        r_positions_a[instance_index].append(transform * component_positions[i]);
      }
      else {
        r_positions_b[instance_index].append(transform * component_positions[i]);
      }
    }
  }
}

static void get_positions_from_instances(Span<GeometryInstanceGroup> set_groups,
                                         const StringRef mask_attribute_name,
                                         MutableSpan<Vector<float3>> r_positions_a,
                                         MutableSpan<Vector<float3>> r_positions_b)
{
  int instance_index = 0;
  for (const GeometryInstanceGroup &set_group : set_groups) {
    const GeometrySet &set = set_group.geometry_set;

    MutableSpan<Vector<float3>> set_instance_positions_a = r_positions_a.slice(
        instance_index, set_group.transforms.size());
    MutableSpan<Vector<float3>> set_instance_positions_b = r_positions_b.slice(
        instance_index, set_group.transforms.size());

    if (set.has<PointCloudComponent>()) {
      gather_positions_from_component_instances(*set.get_component_for_read<PointCloudComponent>(),
                                                mask_attribute_name,
                                                set_group.transforms,
                                                set_instance_positions_a,
                                                set_instance_positions_b);
    }
    if (set.has<MeshComponent>()) {
      gather_positions_from_component_instances(*set.get_component_for_read<MeshComponent>(),
                                                mask_attribute_name,
                                                set_group.transforms,
                                                set_instance_positions_a,
                                                set_instance_positions_b);
    }

    instance_index += set_group.transforms.size();
  }
}

static PointCloud *create_point_cloud(Span<Vector<float3>> positions)
{
  int points_len = 0;
  for (const Vector<float3> &instance_positions : positions) {
    points_len += instance_positions.size();
  }
  PointCloud *pointcloud = BKE_pointcloud_new_nomain(points_len);
  int point_index = 0;
  for (const Vector<float3> &instance_positions : positions) {
    memcpy(pointcloud->co + point_index, positions.data(), sizeof(float3) * positions.size());
    point_index += instance_positions.size();
  }

  return pointcloud;
}

template<typename T>
static void copy_from_span_and_mask(Span<T> span,
                                    Span<bool> mask,
                                    MutableSpan<T> out_span_a,
                                    MutableSpan<T> out_span_b,
                                    int &offset_a,
                                    int &offset_b)
{
  for (const int i : span.index_range()) {
    if (mask[i]) {
      out_span_b[offset_b] = span[i];
      offset_b++;
    }
    else {
      out_span_a[offset_a] = span[i];
      offset_a++;
    }
  }
}

static void copy_attribute_from_component_instances(const GeometryComponent &component,
                                                    const int instances_len,
                                                    const StringRef mask_attribute_name,
                                                    const StringRef attribute_name,
                                                    const CustomDataType data_type,
                                                    fn::GMutableSpan out_data_a,
                                                    fn::GMutableSpan out_data_b,
                                                    int &offset_a,
                                                    int &offset_b)
{
  if (component.attribute_domain_size(ATTR_DOMAIN_POINT) == 0) {
    return;
  }

  const BooleanReadAttribute mask_attribute = component.attribute_get_for_read<bool>(
      mask_attribute_name, ATTR_DOMAIN_POINT, false);

  const ReadAttributePtr attribute = component.attribute_try_get_for_read(
      attribute_name, ATTR_DOMAIN_POINT, data_type);

  Span<bool> masks = mask_attribute.get_span();

  const int start_offset_a = offset_a;
  const int start_offset_b = offset_b;

  attribute_math::convert_to_static_type(data_type, [&](auto dummy) {
    using T = decltype(dummy);
    Span<T> span = attribute->get_span<T>();
    MutableSpan<T> out_span_a = out_data_a.typed<T>();
    MutableSpan<T> out_span_b = out_data_b.typed<T>();
    copy_from_span_and_mask(span, masks, out_span_a, out_span_b, offset_a, offset_b);
    const int copied_len_a = offset_a - start_offset_a;
    const int copied_len_b = offset_b - start_offset_b;
    for (int i = 1; i < instances_len; i++) {
      memcpy(out_span_a.data() + offset_a,
             out_span_a.data() + start_offset_a,
             sizeof(T) * copied_len_a);
      memcpy(out_span_b.data() + offset_b,
             out_span_b.data() + start_offset_b,
             sizeof(T) * copied_len_b);
      offset_a += copied_len_a;
      offset_b += copied_len_b;
    }
  });
}

static void copy_attributes_to_output(Span<GeometryInstanceGroup> set_groups,
                                      Map<std::string, AttributeKind> &result_attributes_info,
                                      const StringRef mask_attribute_name,
                                      PointCloudComponent &out_component_a,
                                      PointCloudComponent &out_component_b)
{
  for (Map<std::string, AttributeKind>::Item entry : result_attributes_info.items()) {
    const StringRef attribute_name = entry.key;
    /* The output domain is always #ATTR_DOMAIN_POINT, since we are creating a point cloud. */
    const CustomDataType output_data_type = entry.value.data_type;

    OutputAttributePtr attribute_out_a = out_component_a.attribute_try_get_for_output(
        attribute_name, ATTR_DOMAIN_POINT, output_data_type);
    OutputAttributePtr attribute_out_b = out_component_b.attribute_try_get_for_output(
        attribute_name, ATTR_DOMAIN_POINT, output_data_type);
    BLI_assert(attribute_out_a && attribute_out_b);
    if (!attribute_out_a || attribute_out_b) {
      continue;
    }

    fn::GMutableSpan out_span_a = attribute_out_a->get_span_for_write_only();
    fn::GMutableSpan out_span_b = attribute_out_b->get_span_for_write_only();

    int offset_a = 0;
    int offset_b = 0;
    for (const GeometryInstanceGroup &set_group : set_groups) {
      const GeometrySet &set = set_group.geometry_set;

      if (set.has<PointCloudComponent>()) {
        copy_attribute_from_component_instances(*set.get_component_for_read<PointCloudComponent>(),
                                                set_group.transforms.size(),
                                                mask_attribute_name,
                                                attribute_name,
                                                output_data_type,
                                                out_span_a,
                                                out_span_b,
                                                offset_a,
                                                offset_b);
      }
      if (set.has<MeshComponent>()) {
        copy_attribute_from_component_instances(*set.get_component_for_read<MeshComponent>(),
                                                set_group.transforms.size(),
                                                mask_attribute_name,
                                                attribute_name,
                                                output_data_type,
                                                out_span_a,
                                                out_span_b,
                                                offset_a,
                                                offset_b);
      }
    }
  }
}

static void geo_node_point_separate_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");
  const std::string mask_attribute_name = params.extract_input<std::string>("Mask");
  if (mask_attribute_name.empty()) {
    params.set_output("Geometry 1", std::move(GeometrySet()));
    params.set_output("Geometry 2", std::move(GeometrySet()));
  }

  Vector<GeometryInstanceGroup> set_groups = bke::geometry_set_gather_instances(geometry_set);

  /* Remove any set inputs that don't contain points, to avoid checking later on. */
  for (int i = set_groups.size() - 1; i >= 0; i--) {
    const GeometrySet &set = set_groups[i].geometry_set;
    if (!set.has_mesh() && !set.has_pointcloud()) {
      set_groups.remove_and_reorder(i);
    }
  }

  int instances_len = 0;
  for (const GeometryInstanceGroup &set_group : set_groups) {
    instances_len += set_group.transforms.size();
  }

  Array<Vector<float3>> positions_a(instances_len);
  Array<Vector<float3>> positions_b(instances_len);
  get_positions_from_instances(set_groups, mask_attribute_name, positions_a, positions_b);

  GeometrySet out_set_a = GeometrySet::create_with_pointcloud(create_point_cloud(positions_a));
  GeometrySet out_set_b = GeometrySet::create_with_pointcloud(create_point_cloud(positions_b));

  Map<std::string, AttributeKind> result_attributes_info;
  bke::gather_attribute_info(result_attributes_info,
                             {GEO_COMPONENT_TYPE_MESH, GEO_COMPONENT_TYPE_POINT_CLOUD},
                             set_groups,
                             {"position"});

  copy_attributes_to_output(set_groups,
                            result_attributes_info,
                            mask_attribute_name,
                            out_set_a.get_component_for_write<PointCloudComponent>(),
                            out_set_b.get_component_for_write<PointCloudComponent>());

  params.set_output("Geometry 1", std::move(out_set_a));
  params.set_output("Geometry 2", std::move(out_set_b));
}
}  // namespace blender::nodes

void register_node_type_geo_point_separate()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_POINT_SEPARATE, "Point Separate", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, geo_node_point_instance_in, geo_node_point_instance_out);
  ntype.geometry_node_execute = blender::nodes::geo_node_point_separate_exec;
  nodeRegisterType(&ntype);
}
