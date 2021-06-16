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

#include "node_geometry_util_interp.hh"

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"

#include "BKE_mesh_runtime.h"
#include "BKE_mesh_sample.hh"

#include "NOD_geometry_exec.hh"

namespace blender::nodes {

Span<MLoopTri> get_mesh_looptris(const Mesh &mesh)
{
  /* This only updates a cache and can be considered to be logically const. */
  const MLoopTri *looptris = BKE_mesh_runtime_looptri_ensure(const_cast<Mesh *>(&mesh));
  const int looptris_len = BKE_mesh_runtime_looptri_len(&mesh);
  return {looptris, looptris_len};
}

AttributeInterpolator::AttributeInterpolator(const Mesh *mesh,
                                             const Span<float3> positions,
                                             const Span<int> looptri_indices)
    : mesh_(mesh), positions_(positions), looptri_indices_(looptri_indices)
{
  BLI_assert(positions.size() == looptri_indices.size());
}

Span<float3> AttributeInterpolator::ensure_barycentric_coords()
{
  if (!bary_coords_.is_empty()) {
    BLI_assert(bary_coords_.size() == positions_.size());
    return bary_coords_;
  }
  bary_coords_.reinitialize(positions_.size());

  Span<MLoopTri> looptris = get_mesh_looptris(*mesh_);

  for (const int i : bary_coords_.index_range()) {
    const int looptri_index = looptri_indices_[i];
    const MLoopTri &looptri = looptris[looptri_index];

    const int v0_index = mesh_->mloop[looptri.tri[0]].v;
    const int v1_index = mesh_->mloop[looptri.tri[1]].v;
    const int v2_index = mesh_->mloop[looptri.tri[2]].v;

    interp_weights_tri_v3(bary_coords_[i],
                          mesh_->mvert[v0_index].co,
                          mesh_->mvert[v1_index].co,
                          mesh_->mvert[v2_index].co,
                          positions_[i]);
  }
  return bary_coords_;
}

Span<float3> AttributeInterpolator::ensure_nearest_weights()
{
  if (!nearest_weights_.is_empty()) {
    BLI_assert(nearest_weights_.size() == positions_.size());
    return nearest_weights_;
  }
  nearest_weights_.reinitialize(positions_.size());

  Span<MLoopTri> looptris = get_mesh_looptris(*mesh_);

  for (const int i : nearest_weights_.index_range()) {
    const int looptri_index = looptri_indices_[i];
    const MLoopTri &looptri = looptris[looptri_index];

    const int v0_index = mesh_->mloop[looptri.tri[0]].v;
    const int v1_index = mesh_->mloop[looptri.tri[1]].v;
    const int v2_index = mesh_->mloop[looptri.tri[2]].v;

    const float d0 = len_squared_v3v3(positions_[i], mesh_->mvert[v0_index].co);
    const float d1 = len_squared_v3v3(positions_[i], mesh_->mvert[v1_index].co);
    const float d2 = len_squared_v3v3(positions_[i], mesh_->mvert[v2_index].co);

    nearest_weights_[i] = MIN3_PAIR(d0, d1, d2, float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1));
  }
  return nearest_weights_;
}

void AttributeInterpolator::sample_attribute(const ReadAttributeLookup &src_attribute,
                                             OutputAttribute &dst_attribute,
                                             eAttributeMapMode mode)
{
  if (looptri_indices_.is_empty()) {
    return;
  }

  if (!src_attribute || !dst_attribute) {
    return;
  }
  const GVArray &src_varray = *src_attribute.varray;
  GMutableSpan dst_span = dst_attribute.as_span();
  if (src_varray.is_empty() || dst_span.is_empty()) {
    return;
  }

  /* Compute barycentric coordinates only when they are needed. */
  Span<float3> weights;
  if (ELEM(src_attribute.domain, ATTR_DOMAIN_POINT, ATTR_DOMAIN_CORNER)) {
    switch (mode) {
      case eAttributeMapMode::INTERPOLATED:
        weights = ensure_barycentric_coords();
        break;
      case eAttributeMapMode::NEAREST:
        weights = ensure_nearest_weights();
        break;
    }
  }

  /* Interpolate the source attributes on the surface. */
  switch (src_attribute.domain) {
    case ATTR_DOMAIN_POINT: {
      bke::mesh_surface_sample::sample_point_attribute(
          *mesh_, looptri_indices_, weights, src_varray, dst_span);
      break;
    }
    case ATTR_DOMAIN_FACE: {
      bke::mesh_surface_sample::sample_face_attribute(
          *mesh_, looptri_indices_, src_varray, dst_span);
      break;
    }
    case ATTR_DOMAIN_CORNER: {
      bke::mesh_surface_sample::sample_corner_attribute(
          *mesh_, looptri_indices_, weights, src_varray, dst_span);
      break;
    }
    case ATTR_DOMAIN_EDGE: {
      /* Not yet supported. */
      break;
    }
    default: {
      BLI_assert_unreachable();
      break;
    }
  }
}

}  // namespace blender::nodes
