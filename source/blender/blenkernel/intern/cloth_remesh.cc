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

#include "DNA_cloth_types.h"
#include "DNA_mesh_types.h"
#include "DNA_modifier_types.h"
#include "DNA_object_types.h"

#include "BLI_assert.h"
#include "BLI_float2x2.hh"
#include "BLI_math_vector.h"
#include "BLI_utildefines.h"

#include "MEM_guardedalloc.h"

#include "BKE_cloth.h"
#include "BKE_cloth_remesh.hh"
#include <limits>

namespace blender::bke::internal {
class NodeData;
class VertData;
class Sizing;

template<typename T> static inline T simple_interp(const T &a, const T &b)
{
  return (a + b) * 0.5;
}

class NodeData {
  ClothVertex cloth_node_data; /* The cloth simulation calls it
                                * Vertex, internal::Mesh calls it Node */
 public:
  NodeData(const ClothVertex &cloth_node_data) : cloth_node_data(cloth_node_data)
  {
  }

  NodeData(ClothVertex &&cloth_node_data) : cloth_node_data(cloth_node_data)
  {
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

  Sizing interp(const Sizing &other) const
  {
    return Sizing(this->m.linear_blend(other.get_m(), 0.5));
  }
};

enum VertFlags {
  VERT_NONE = 0,
  VERT_SELECTED = 1 << 0,
};

class VertData {
  Sizing sizing; /* in [1], this is the "sizing" of the verts */
  int flag;

 public:
  VertData(Sizing sizing) : sizing(sizing)
  {
    this->flag = VERT_NONE;
  }

  const auto &get_sizing() const
  {
    return this->sizing;
  }

  void set_sizing(Sizing sizing)
  {
    this->sizing = sizing;
  }

  const auto &get_flag() const
  {
    return this->flag;
  }

  auto &get_flag_mut()
  {
    return this->flag;
  }

  VertData interp(const VertData &other) const
  {
    VertData res(this->get_sizing().interp(other.get_sizing()));
    res.flag = VERT_NONE;
    return res;
  }
};

class EdgeData {
  float size; /* from [1], the size calculated by the `Vert`s of the
               * `Edge` */

 public:
  EdgeData(float size) : size(size)
  {
  }

  const auto &get_size() const
  {
    return this->size;
  }

  void set_sizing(float size)
  {
    this->size = size;
  }

  EdgeData interp(const EdgeData &UNUSED(other)) const
  {
    return EdgeData(std::numeric_limits<float>::signaling_NaN());
  }
};

class AdaptiveMesh : public Mesh<NodeData, VertData, EdgeData, internal::EmptyExtraData> {
 public:
  void set_nodes_extra_data(const Cloth &cloth)
  {
    /* The layout of the `this->get_nodes()` and `cloth.verts` should
     * be the same, so just directly copy it over */
    BLI_assert(cloth.mvert_num == this->get_nodes().size());

    auto i = 0;
    for (auto &node : this->get_nodes_mut()) {
      node.set_extra_data(NodeData(cloth.verts[i]));
      i++;
    }
  }

  /**
   * Sets the "size" of the `Edge`s of the mesh by running
   * `get_edge_size_sq()` on the `Sizing` stored in the `Vert`s of the
   * `Edge`.
   *
   * `Sizing` has to be set for the `Vert`s before calling this function.
   */
  void set_edge_sizes()
  {
    for (auto &edge : this->get_edges_mut()) {
      const auto [v1, v2] = this->get_checked_verts_of_edge(edge, false);
      const auto &v1_uv = v1.get_uv();
      const auto &v2_uv = v2.get_uv();
      const auto v1_sizing = v1.get_checked_extra_data().get_sizing();
      const auto v2_sizing = v2.get_checked_extra_data().get_sizing();

      auto edge_size = v1_sizing.get_edge_size_sq(v2_sizing, v1_uv, v2_uv);
      auto op_edge_data = edge.get_extra_data_mut();
      if (op_edge_data) {
        auto &edge_data = edge.get_checked_extra_data_mut();
        edge_data.set_sizing(edge_size);
      }
      else {
        edge.set_extra_data(EdgeData(edge_size));
      }
    }
  }

  /**
   * Splits edges whose "size" is greater than 1.0
   *
   * Based on [1]
   *
   * Here "size" is determined by `Sizing` stores in `Vert`s of the
   * `Edge`, using the function `Sizing::get_edge_size_sq()`.
   */
  void split_edges()
  {
    auto splittable_edges_set = this->get_splittable_edge_indices_set();
    do {
      for (const auto &edge_index : splittable_edges_set) {
        auto &edge = this->get_checked_edge(edge_index);
        this->split_edge_triangulate(edge.get_self_index(), true);

        /* TODO(ish): Need to flip edges of those faces that have been
         * affected by the split edge operation. */
      }

      splittable_edges_set = this->get_splittable_edge_indices_set();
    } while (splittable_edges_set.size() == 0);
  }

 private:
  /**
   * Gets the maximal independent set of splittable edge indices in
   * the `AdaptiveMesh`.
   *
   * Reference [1]
   */
  blender::Vector<EdgeIndex> get_splittable_edge_indices_set()
  {
    /* Deselect all verts */
    for (auto &vert : this->get_verts_mut()) {
      auto &vert_data = vert.get_checked_extra_data_mut();
      auto &flag = vert_data.get_flag_mut();
      flag &= ~VERT_SELECTED;
    }

    blender::Vector<EdgeIndex> splittable_edge_indices;
    /* It is assumed that the edges sizes have been computed earlier
     * and stored in the extra data of the edges */
    for (const auto &edge : this->get_edges()) {
      auto [v1, v2] = this->get_checked_verts_of_edge(edge, false);
      if (v1.get_checked_extra_data().get_flag() & VERT_SELECTED ||
          v2.get_checked_extra_data().get_flag() & VERT_SELECTED) {
        continue;
      }
      const auto &edge_data = edge.get_checked_extra_data();
      auto edge_size = edge_data.get_size();
      if (edge_size > 1.0) {
        splittable_edge_indices.append(edge.get_self_index());
        auto &v1_data = v1.get_checked_extra_data_mut();
        auto &v1_flag = v1_data.get_flag_mut();
        v1_flag |= VERT_SELECTED;
        auto &v2_data = v2.get_checked_extra_data_mut();
        auto &v2_flag = v2_data.get_flag_mut();
        v2_flag |= VERT_SELECTED;
      }
    }

    return splittable_edge_indices;
  }
};

static void cloth_delete_verts(Cloth &cloth)
{
  BLI_assert(cloth.verts);
  MEM_freeN(cloth.verts);
  cloth.verts = nullptr;
}

static void cloth_set_verts_from_adaptive_mesh(Cloth &cloth, const AdaptiveMesh &mesh)
{
  /* caller should have deleted the verts earlier */
  BLI_assert(cloth.verts == nullptr);

  cloth.verts = static_cast<ClothVertex *>(
      MEM_callocN(sizeof(ClothVertex) * mesh.get_nodes().size(), __func__));

  auto i = 0;
  for (const auto &node : mesh.get_nodes()) {
    const auto &op_extra_data = node.get_extra_data();
    BLI_assert(op_extra_data);
    cloth.verts[i] = op_extra_data.value().get_cloth_node_data();
    i++;
  }
}

}  // namespace blender::bke::internal

namespace blender::bke {

Mesh *BKE_cloth_remesh(Object *ob, ClothModifierData *clmd, Mesh *mesh)
{
  auto *cloth_to_object_res = cloth_to_object(ob, clmd, mesh, false);
  BLI_assert(cloth_to_object_res == nullptr);

  internal::MeshIO meshio_input;
  meshio_input.read(mesh);

  internal::AdaptiveMesh adaptive_mesh;
  adaptive_mesh.read(meshio_input);

  {
    adaptive_mesh.set_nodes_extra_data(*clmd->clothObject);
    internal::cloth_delete_verts(*clmd->clothObject);
  }

  internal::cloth_set_verts_from_adaptive_mesh(*clmd->clothObject, adaptive_mesh);

  auto meshio_output = adaptive_mesh.write();
  return meshio_output.write();
}

}  // namespace blender::bke
