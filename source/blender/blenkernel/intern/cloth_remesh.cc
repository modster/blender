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
 *
 * The Original Code is Copyright (C) Blender Foundation
 * All rights reserved.
 */

/** \file
 * \ingroup bke
 */

#include "BLI_math_vector.h"
#include "DNA_cloth_types.h"
#include "DNA_mesh_types.h"
#include "DNA_object_types.h"

#include "BLI_float2x2.hh"
#include "BLI_utildefines.h"

#include "BKE_cloth.h"
#include "BKE_cloth_remesh.hh"

#include <cstdio>

namespace blender::bke {
class NodeData;
class VertData;
class Sizing;

using AdaptiveMesh =
    internal::Mesh<NodeData, VertData, internal::EmptyExtraData, internal::EmptyExtraData>;

template<typename T> static inline T simple_interp(const T &a, const T &b)
{
  return (a + b) * 0.5;
}

class NodeData {
  ClothVertex cloth_node_data; /* The cloth simulation calls it
                                * Vertex, internal::Mesh calls it Node */
 public:
  NodeData() = default;
  NodeData(ClothVertex &&cloth_node_data) : cloth_node_data(cloth_node_data)
  {
  }

  void set_cloth_node_data(ClothVertex &&cloth_node_data)
  {
    this->cloth_node_data = std::move(cloth_node_data);
  }

  const auto &get_cloth_node_data() const
  {
    return this->cloth_node_data;
  }

  NodeData interp(const NodeData &other) const
  {
    {
      /* This check is to ensure that any new element added to
       * ClothVertex is also updated here. After adding the
       * interpolated value for the element (if needed), set the
       * correct sizeof(ClothVertex) in the assertion below. */
      BLI_assert(sizeof(ClothVertex) == 125);
    }

    ClothVertex cn;
    /* TODO(ish): figure out how to handle the flags */
    /* cn.flags; */
    interp_v3_v3v3(cn.v, this->cloth_node_data.v, other.cloth_node_data.v, 0.5);
    interp_v3_v3v3(cn.xconst, this->cloth_node_data.xconst, other.cloth_node_data.xconst, 0.5);
    interp_v3_v3v3(cn.x, this->cloth_node_data.x, other.cloth_node_data.x, 0.5);
    interp_v3_v3v3(cn.xold, this->cloth_node_data.xold, other.cloth_node_data.xold, 0.5);
    interp_v3_v3v3(cn.tx, this->cloth_node_data.tx, other.cloth_node_data.tx, 0.5);
    interp_v3_v3v3(cn.txold, this->cloth_node_data.txold, other.cloth_node_data.txold, 0.5);
    interp_v3_v3v3(cn.tv, this->cloth_node_data.tv, other.cloth_node_data.tv, 0.5);
    cn.mass = simple_interp(this->cloth_node_data.mass, other.cloth_node_data.mass);
    cn.goal = simple_interp(this->cloth_node_data.goal, other.cloth_node_data.goal);
    interp_v3_v3v3(cn.impulse, this->cloth_node_data.impulse, other.cloth_node_data.impulse, 0.5);
    interp_v3_v3v3(cn.xrest, this->cloth_node_data.xrest, other.cloth_node_data.xrest, 0.5);
    interp_v3_v3v3(cn.dcvel, this->cloth_node_data.dcvel, other.cloth_node_data.dcvel, 0.5);
    /* TODO(ish): these might need to be set else where */
    {
      cn.impulse_count = simple_interp(this->cloth_node_data.impulse_count,
                                       other.cloth_node_data.impulse_count);
      cn.avg_spring_len = simple_interp(this->cloth_node_data.avg_spring_len,
                                        other.cloth_node_data.avg_spring_len);
      cn.struct_stiff = simple_interp(this->cloth_node_data.struct_stiff,
                                      other.cloth_node_data.struct_stiff);
      cn.bend_stiff = simple_interp(this->cloth_node_data.bend_stiff,
                                    other.cloth_node_data.bend_stiff);
      cn.shear_stiff = simple_interp(this->cloth_node_data.shear_stiff,
                                     other.cloth_node_data.shear_stiff);
      cn.spring_count = simple_interp(this->cloth_node_data.spring_count,
                                      other.cloth_node_data.spring_count);
      cn.shrink_factor = simple_interp(this->cloth_node_data.shrink_factor,
                                       other.cloth_node_data.shrink_factor);
      cn.internal_stiff = simple_interp(this->cloth_node_data.internal_stiff,
                                        other.cloth_node_data.internal_stiff);
      cn.pressure_factor = simple_interp(this->cloth_node_data.pressure_factor,
                                         other.cloth_node_data.pressure_factor);
    }
    return NodeData(std::move(cn));
  }
};

class Sizing {
  float2x2 m; /* in [1], the "sizing" tensor field given as `M` */

 public:
  Sizing(float2x2 &&m) : m(m)
  {
  }

  const auto &get_m() const
  {
    return this->m;
  }

  /**
   * Returns the squared edge size
   *
   * @param other Sizing of the other vertex along the edge
   * @param u_i Material space coorindates (uv coords) of the `this`
   * vert
   * @param u_j Material space coorindates (uv coords) of the `other`
   * vert
   */
  float get_edge_size_sq(const Sizing &other, const float2 &u_i, const float2 &u_j) const
  {
    /* The edge size is given by
     * s(i, j)^2 = transpose(uv_ij) * ((m_i + m_j) / 2) * uv_ij
     *
     * This means that the "size" requires a sqrt but since the "size" of
     * the edge is generally used only to check against 1.0, it is more
     * performant to not do the sqrt operation.
     */
    const float2 u_ij = u_i - u_j;

    const float2x2 m_avg = (this->m + other.m) * 0.5;

    return float2::dot(u_ij, m_avg * u_ij);
  }
};

class VertData {
  Sizing sizing; /* in [1], this is the "sizing" of the verts */

 public:
  VertData(Sizing &&sizing) : sizing(sizing)
  {
  }

  const auto &get_sizing() const
  {
    return this->sizing;
  }
};

Mesh *BKE_cloth_remesh(Object *ob, ClothModifierData *clmd, Mesh *mesh)
{
  auto *cloth_to_object_res = cloth_to_object(ob, clmd, mesh, false);
  BLI_assert(cloth_to_object_res == nullptr);

  internal::MeshIO meshio_input;
  meshio_input.read(mesh);

  AdaptiveMesh adaptive_mesh;
  adaptive_mesh.read(meshio_input);

  auto meshio_output = adaptive_mesh.write();
  return meshio_output.write();
}

}  // namespace blender::bke
