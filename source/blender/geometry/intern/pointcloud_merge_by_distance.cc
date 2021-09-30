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
                            MutableSpan<bool> merge_map,
                            const float merge_threshold,
                            Span<bool> selection)
{
  KDTree_3d *kdtree = build_kdtree(positions, selection);

  for (int i : positions.index_range()) {
    struct CallbackData {
      int index;
      MutableSpan<bool> merge_map;
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
              !callback_data.merge_map[source_point_index] &&
              !callback_data.merge_map[target_point_index] &&
              callback_data.selection[source_point_index] &&
              callback_data.selection[target_point_index]) {
            callback_data.merge_map[source_point_index] = true;
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
  const PointCloud &original_src_pointcloud = *pointcloud_component.get_for_read();
  Array<bool> merge_map(original_src_pointcloud.totpoint, false);
  Span<float3> positions((const float3 *)original_src_pointcloud.co,
                         original_src_pointcloud.totpoint);

  build_merge_map(positions, merge_map, merge_threshold, selection);

  Vector<int64_t> copy_mask_vector;
  for (const int i : positions.index_range()) {
    if (!merge_map[i]) {
      copy_mask_vector.append(i);
    }
  }
  IndexMask copyMask(copy_mask_vector);

  PointCloudComponent *src_pointcloud_component = (PointCloudComponent *)
                                                      pointcloud_component.copy();
  const PointCloud &src_pointcloud = *src_pointcloud_component->get_for_read();

  PointCloud *result = BKE_pointcloud_new_nomain(copyMask.size());
  pointcloud_component.replace(result);

  for (const int i : copyMask.index_range()) {
    copy_v3_v3(result->co[i], src_pointcloud.co[copyMask[i]]);
    result->radius[i] = src_pointcloud.radius[copyMask[i]];
  }
  src_pointcloud_component->attribute_foreach(
      [&](const bke::AttributeIDRef &attribute_id, const AttributeMetaData &meta_data) {
        fn::GVArrayPtr read_attribute = src_pointcloud_component->attribute_get_for_read(
            attribute_id, meta_data.domain, meta_data.data_type);

        if (pointcloud_component.attribute_try_create(
                attribute_id, meta_data.domain, meta_data.data_type, AttributeInitDefault())) {

          bke::OutputAttribute target_attribute =
              pointcloud_component.attribute_try_get_for_output_only(
                  attribute_id, meta_data.domain, meta_data.data_type);

          fn::GMutableSpan dst_span = target_attribute.as_span();
          fn::GSpan src_span = read_attribute->get_internal_span();
          for (const int i : copyMask.index_range()) {
            const fn::CPPType *type = bke::custom_data_type_to_cpp_type(meta_data.data_type);
            type->copy_assign(src_span[copyMask[i]], dst_span[i]);
          }

          target_attribute.save();
        }
        return true;
      });

  return result;
}

}  // namespace blender::geometry
