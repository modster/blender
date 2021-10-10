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
#include "BKE_geometry_set.hh"
#include "BKE_pointcloud.h"

#include "BLI_array.hh"
#include "BLI_float3.hh"
#include "BLI_kdtree.h"
#include "BLI_multi_value_map.hh"
#include "BLI_span.hh"
#include "BLI_vector.hh"

#include "DNA_pointcloud_types.h"

#include "GEO_pointcloud_merge_by_distance.hh"

#include "FN_generic_span.hh"

using blender::MutableSpan;

namespace blender::geometry {
const static int POINT_NOT_MERGED = -1;

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
              callback_data.merge_map[source_point_index] == POINT_NOT_MERGED &&
              callback_data.merge_map[target_point_index] == POINT_NOT_MERGED &&
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
  Array<int> merge_map(src_pointcloud.totpoint, POINT_NOT_MERGED);
  Span<float3> positions((const float3 *)src_pointcloud.co, src_pointcloud.totpoint);

  build_merge_map(positions, merge_map, merge_threshold, selection);

  MultiValueMap<int, int> copy_map;
  for (const int i : merge_map.index_range()) {
    if (merge_map[i] != POINT_NOT_MERGED) {
      copy_map.add(merge_map[i], i);
    }
  }

  PointCloud *pointcloud = BKE_pointcloud_new_nomain(copy_map.size());
  PointCloudComponent dst_component;
  dst_component.replace(pointcloud, GeometryOwnershipType::Editable);

  pointcloud_component.attribute_foreach(
      [&](const bke::AttributeIDRef &attribute_id, const AttributeMetaData &meta_data) {
        fn::GVArrayPtr read_attribute = pointcloud_component.attribute_get_for_read(
            attribute_id, meta_data.domain, meta_data.data_type);

        if (!dst_component.attribute_exists(attribute_id) &&
            !dst_component.attribute_try_create(
                attribute_id, meta_data.domain, meta_data.data_type, AttributeInitDefault())) {
          return true;
        }

        bke::OutputAttribute target_attribute = dst_component.attribute_try_get_for_output_only(
            attribute_id, meta_data.domain, meta_data.data_type);

        blender::attribute_math::convert_to_static_type(meta_data.data_type, [&](auto dummy) {
          using T = decltype(dummy);
          const fn::GVArray_Typed<T> src_span = read_attribute->typed<T>();

          attribute_math::DefaultMixer<T> mixer(target_attribute.as_span<T>());

          int index_new = 0;
          for (const int index_old : copy_map.keys()) {
            Span<int> merged_points = copy_map.lookup(index_old);
            if (merged_points.size() > 0) {
              float weight = 1.0f / (float(merged_points.size() + 1.0f));
              mixer.mix_in(index_new, src_span[index_old], weight);
              for (const int j : merged_points) {
                mixer.mix_in(index_new, src_span[j], weight);
              }
            }
            index_new++;
          }
          mixer.finalize();
        });

        target_attribute.save();
        return true;
      });

  return pointcloud;
}

}  // namespace blender::geometry
