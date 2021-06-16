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

#pragma once

#include "BKE_attribute_access.hh"

struct Mesh;
struct MLoopTri;

namespace blender::nodes {

Span<MLoopTri> get_mesh_looptris(const Mesh &mesh);

enum class eAttributeMapMode {
  INTERPOLATED,
  NEAREST,
};

/**
 * A utility class that performs attribute interpolation from a source mesh.
 *
 * The interpolator is only valid as long as the mesh is valid.
 * Barycentric weights are needed when interpolating point or corner domain attributes,
 * these are computed lazily when needed and re-used.
 */
class AttributeInterpolator {
 private:
  const Mesh *mesh_;
  const Span<float3> positions_;
  const Span<int> looptri_indices_;

  Array<float3> bary_coords_;
  Array<float3> nearest_weights_;

 public:
  AttributeInterpolator(const Mesh *mesh,
                        const Span<float3> positions,
                        const Span<int> looptri_indices);

  void sample_attribute(const blender::bke::ReadAttributeLookup &src_attribute,
                        blender::bke::OutputAttribute &dst_attribute,
                        eAttributeMapMode mode);

 protected:
  Span<float3> ensure_barycentric_coords();
  Span<float3> ensure_nearest_weights();
};

}  // namespace blender::nodes
