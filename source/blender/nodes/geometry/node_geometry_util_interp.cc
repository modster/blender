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

void AttributeInterpolator::compute_barycentric_coords()
{
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
}

void AttributeInterpolator::interpolate_attribute(const ReadAttributeLookup &src_attribute,
                                                  OutputAttribute &dst_attribute)
{
  if (!src_attribute || !dst_attribute) {
    return;
  }
  const GVArray &src_varray = *src_attribute.varray;
  GMutableSpan dst_span = dst_attribute.as_span();
  if (src_varray.is_empty() || dst_span.is_empty()) {
    return;
  }

  if (looptri_indices_.is_empty()) {
  }

  /* Compute barycentric coordinates only when they are needed. */
  if (bary_coords_.is_empty() &&
      ELEM(src_attribute.domain, ATTR_DOMAIN_POINT, ATTR_DOMAIN_CORNER)) {
    compute_barycentric_coords();
  }

  /* Interpolate the source attributes on the surface. */
  switch (src_attribute.domain) {
    case ATTR_DOMAIN_POINT: {
      bke::mesh_surface_sample::sample_point_attribute(
          *mesh_, looptri_indices_, bary_coords_, src_varray, dst_span);
      break;
    }
    case ATTR_DOMAIN_FACE: {
      bke::mesh_surface_sample::sample_face_attribute(
          *mesh_, looptri_indices_, src_varray, dst_span);
      break;
    }
    case ATTR_DOMAIN_CORNER: {
      bke::mesh_surface_sample::sample_corner_attribute(
          *mesh_, looptri_indices_, bary_coords_, src_varray, dst_span);
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
