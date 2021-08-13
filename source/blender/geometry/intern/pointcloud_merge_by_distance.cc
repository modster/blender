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

#include "BLI_array.hh"
#include "BLI_float3.hh"
#include "BLI_kdtree.h"
#include "BLI_span.hh"
#include "BLI_vector.hh"

#include "DNA_pointcloud_types.h"

#include "GEO_pointcloud_merge_by_distance.h" /* Own include. */

#include "FN_generic_span.hh"

using blender::Array;
using blender::float3;
using blender::Span;
using blender::Vector;

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

static void build_merge_map(Span<float3> &positions,
                            Array<bool> &merge_map,
                            int &total_merge_operations,
                            const float merge_threshold,
                            Span<bool> selection)
{
  KDTree_3d *kdtree = build_kdtree(positions, selection);

  for (int i : positions.index_range()) {
    struct CallbackData {
      int index;
      int &total_merge_operations;
      Array<bool> &merge_map;
      Span<bool> selection;
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
              !callback_data.merge_map[source_vertex_index] &&
              !callback_data.merge_map[target_vertex_index] &&
              callback_data.selection[source_vertex_index] &&
              callback_data.selection[target_vertex_index]) {
            callback_data.merge_map[source_vertex_index] = true;
            callback_data.total_merge_operations++;
          }
          return true;
        },
        &callback_data);
  }

  BLI_kdtree_3d_free(kdtree);
}

PointCloud *merge_by_distance_pointcloud(const PointCloud &point_cloud,
                                         const float merge_threshold,
                                         Span<bool> selection)
{

  Array<bool> merge_map(point_cloud.totpoint, false);
  Span<float3> positions((const float3 *)point_cloud.co, point_cloud.totpoint);

  int total_merge_operations = 0;

  BLI_assert(positions.size() == merge_map.size());

  build_merge_map(positions, merge_map, total_merge_operations, merge_threshold, selection);

  PointCloud *result = BKE_pointcloud_new_nomain(positions.size() - total_merge_operations);
  int offset = 0;
  for (const int i : positions.index_range()) {
    /* Only copy the unmerged points to new pointcloud. */
    if (!merge_map[i]) {
      copy_v3_v3(result->co[offset], positions[i]);
      result->radius[offset] = point_cloud.radius[i];
      offset++;
    }
  }

  return result;
}