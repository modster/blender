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

#include "BKE_geometry_set.hh"
#include "BKE_pointcloud.h"
#include <BKE_attribute_math.hh>

#include "BLI_array.hh"
#include "BLI_float3.hh"
#include "BLI_kdtree.h"
#include "BLI_span.hh"
#include "BLI_vector.hh"

#include "DNA_pointcloud_types.h"

#include "GEO_pointcloud_merge_by_distance.hh"

#include "FN_generic_span.hh"

using blender::MutableSpan;

namespace blender::geometry {

static KDTree_3d *build_kdtree(Span<float3> positions, Span<bool> selection)
{
  BLI_assert(positions.size() == selection.size());

  KDTree_3d *kdtree = BLI_kdtree_3d_new(selection.size());

  for (const int i : positions.index_range()) {
    if (selection[i]) {
      BLI_kdtree_3d_insert(kdtree, i, positions[i]);
    }
  }

  BLI_kdtree_3d_balance(kdtree);
  return kdtree;
}

static void build_merge_map(Span<float3> positions,
                            MutableSpan<int> merge_map,
                            const float merge_threshold,
                            Span<bool> selection)
{
  KDTree_3d *kdtree = build_kdtree(positions, selection);

  for (int i : positions.index_range()) {
    struct CallbackData {
      int index;
      MutableSpan<int> merge_map;
      Span<bool> selection;
    } callback_data = {i, merge_map, selection};

    BLI_kdtree_3d_range_search_cb(
        kdtree,
        positions[i],
        merge_threshold,
        [](void *user_data,
           int source_point_index,
           const float *UNUSED(co),
           float UNUSED(distance_squared)) {
          CallbackData &callback_data = *static_cast<CallbackData *>(user_data);
          int target_point_index = callback_data.index;
          if (source_point_index != target_point_index &&
              callback_data.merge_map[source_point_index] == -1 &&
              callback_data.merge_map[target_point_index] == -1 &&
              callback_data.selection[source_point_index] &&
              callback_data.selection[target_point_index]) {
            callback_data.merge_map[source_point_index] = target_point_index;
          }
          return true;
        },
        &callback_data);
  }

  BLI_kdtree_3d_free(kdtree);
}

PointCloud *pointcloud_merge_by_distance(PointCloudComponent &pointcloud_component,
                                         const float merge_threshold,
                                         Span<bool> selection)
{
  const PointCloud &src_pointcloud = *pointcloud_component.get_for_read();
  Array<int> merge_map(src_pointcloud.totpoint, -1);
  Span<float3> positions((const float3 *)src_pointcloud.co, src_pointcloud.totpoint);

  build_merge_map(positions, merge_map, merge_threshold, selection);

  Vector<int64_t> copy_mask_vector;
  for (const int i : positions.index_range()) {
    if (merge_map[i] == -1) {
      copy_mask_vector.append(i);
    }
  }
  IndexMask copy_mask(copy_mask_vector);

  PointCloud *pointcloud = BKE_pointcloud_new_nomain(src_pointcloud.totpoint);
  PointCloudComponent dst_component;
  dst_component.replace(pointcloud, GeometryOwnershipType::Editable);

  for (const int i : copy_mask.index_range()) {
    copy_v3_v3(pointcloud->co[i], src_pointcloud.co[copy_mask[i]]);
    pointcloud->radius[i] = src_pointcloud.radius[copy_mask[i]];
  }

  pointcloud_component.attribute_foreach(
      [&](const bke::AttributeIDRef &attribute_id, const AttributeMetaData &meta_data) {
        fn::GVArrayPtr read_attribute = pointcloud_component.attribute_get_for_read(
            attribute_id, meta_data.domain, meta_data.data_type);

        if (dst_component.attribute_try_create(
                attribute_id, meta_data.domain, meta_data.data_type, AttributeInitDefault())) {

          bke::OutputAttribute target_attribute = dst_component.attribute_try_get_for_output_only(
              attribute_id, meta_data.domain, meta_data.data_type);

          blender::attribute_math::convert_to_static_type(meta_data.data_type, [&](auto dummy) {
            using T = decltype(dummy);
            // fn::GVMutableArray_Typed<T> dst_span = target_attribute->typed<T>();
            // const fn::GVArray_Typed<T> src_span = read_attribute->typed<T>();
            // target_attribute.template as_span<>()
            // dst_span->materialize_to_uninitialized(buffer);
            attribute_math::DefaultMixer<T> mixer(target_attribute.as_span<T>());

            // for (const int j : merge_map.index_range()) {
            //   if (merge_map[j] == src_index) {
            //     mixer.mix_in(src_span[j], 0.1f);
            //    }
            // }
            mixer.finalize();
          });

          target_attribute.save();

          //          fn::GMutableSpan dst_span = target_attribute.as_span();
          //          const fn::GSpan src_span = read_attribute->get_internal_span();
          //          for (const int i : copy_mask.index_range()) {
          //            const int src_index = copy_mask[i];
          //            const fn::CPPType *type =
          //            bke::custom_data_type_to_cpp_type(meta_data.data_type);
          //            type->copy_assign(src_span[src_index], dst_span[i]);
          //            target_attribute.save();
          //          }
        }
        return true;
      });

  return pointcloud;
}

}  // namespace blender::geometry
