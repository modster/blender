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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>

#define SHOULD_REMESH_DUMP_FILE 1

namespace blender::bke::internal {
static FilenameGen static_remesh_name_gen("/tmp/static_remesh/remesh", ".mesh");

class ClothNodeData;

template<typename T> class NodeData;
/* TODO(ish): make the other "XData" generic */
class VertData;
class EdgeData;

class Sizing;

template<typename T> static inline T simple_interp(const T &a, const T &b)
{
  return (a + b) * 0.5;
}

template<typename T> class NodeData {
  T extra_data;

 public:
  /* NodeData can be created only if `extra_data` is provided for so
   * it is fine to not store `extra_data` as an optional */
  NodeData(T extra_data) : extra_data(extra_data)
  {
  }

  void set_extra_data(T extra_data)
  {
    this->extra_data = extra_data;
  }

  const auto &get_extra_data() const
  {
    return this->extra_data;
  }

  auto &get_extra_data_mut()
  {
    return this->extra_data;
  }

  NodeData interp(const NodeData &other) const
  {
    return NodeData(this->extra_data.interp(other.get_extra_data()));
  }
};

class ClothNodeData {
  ClothVertex cloth_node_data; /* The cloth simulation calls it
                                * Vertex, internal::Mesh calls it Node */
 public:
  ClothNodeData(const ClothVertex &cloth_node_data) : cloth_node_data(cloth_node_data)
  {
  }

  ClothNodeData(ClothVertex &&cloth_node_data) : cloth_node_data(cloth_node_data)
  {
  }

  const auto &get_cloth_node_data() const
  {
    return this->cloth_node_data;
  }

  ClothNodeData interp(const ClothNodeData &other) const
  {
    {
      /* This check is to ensure that any new element added to
       * ClothVertex is also updated here. After adding the
       * interpolated value for the element (if needed), set the
       * correct sizeof(ClothVertex) in the assertion below. */
      BLI_assert(sizeof(ClothVertex) == 168);
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
    return ClothNodeData(std::move(cn));
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

  friend Sizing operator+(const Sizing &a, const Sizing &b)
  {
    return Sizing(a.get_m() + b.get_m());
  }

  friend Sizing operator*(const Sizing &a, float val)
  {
    return Sizing(a.get_m() * val);
  }

  friend float2 operator*(const Sizing &sizing, const float2 &v)
  {
    return sizing.get_m() * v;
  }
};

enum VertFlags {
  VERT_NONE = 0,
  VERT_SELECTED_FOR_SPLIT = 1 << 0,
  VERT_SELECTED_FOR_FLIP = 1 << 1,
  VERT_PRESERVE = 1 << 2,
};

class VertData {
  Sizing sizing; /* in [1], this is the "sizing" of the verts */
  int flag;

 public:
  VertData() : sizing(float2x2::identity())
  {
  }

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

template<typename T> using AdaptiveNode = Node<NodeData<T>>;
using AdaptiveVert = Vert<VertData>;
using AdaptiveEdge = Edge<EdgeData>;
using AdaptiveFace = Face<internal::EmptyExtraData>;
template<typename T>
using AdaptiveMeshDiff = MeshDiff<NodeData<T>, VertData, EdgeData, internal::EmptyExtraData>;

template<typename END>
class AdaptiveMesh : public Mesh<NodeData<END>, VertData, EdgeData, internal::EmptyExtraData> {
 public:
  float compute_edge_size(const AdaptiveVert &v1, const AdaptiveVert &v2) const
  {
    const auto &v1_uv = v1.get_uv();
    const auto &v2_uv = v2.get_uv();
    const auto v1_sizing = v1.get_checked_extra_data().get_sizing();
    const auto v2_sizing = v2.get_checked_extra_data().get_sizing();

    return v1_sizing.get_edge_size_sq(v2_sizing, v1_uv, v2_uv);
  }

  float compute_edge_size(const AdaptiveEdge &edge) const
  {
    const auto [v1, v2] = this->get_checked_verts_of_edge(edge, false);

    return this->compute_edge_size(v1, v2);
  }

  void edge_set_size(AdaptiveEdge &edge)
  {
    const auto edge_size = this->compute_edge_size(edge);

    auto op_edge_data = edge.get_extra_data_mut();
    if (op_edge_data) {
      auto &edge_data = edge.get_checked_extra_data_mut();
      edge_data.set_sizing(edge_size);
    }
    else {
      edge.set_extra_data(EdgeData(edge_size));
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
      this->edge_set_size(edge);
    }
  }

  /**
   * Marks verts which are on a seam or boundary for preserve
   */
  void mark_verts_for_preserve()
  {
    for (auto &vert : this->get_verts_mut()) {
      if (this->is_vert_on_seam_or_boundary(vert)) {
        auto &op_vert_data = vert.get_extra_data_mut();
        if (op_vert_data) {
          auto &vert_data = op_vert_data.value();
          vert_data.get_flag_mut() |= VERT_PRESERVE;
        }
        else {
          vert.set_extra_data(VertData());
          vert.get_extra_data_mut().value().get_flag_mut() |= VERT_PRESERVE;
        }
      }
    }
  }

  /**
   * Flip edges of the `active_faces` if needed.
   *
   * Updates the active_faces in place
   */
  void flip_edges(blender::Vector<FaceIndex> &active_faces)
  {
    auto max_loop_cycles = active_faces.size() * 3;
    auto loop_cycles_until_now = 0;
    auto flippable_edge_indices_set = this->get_flippable_edge_indices_set(active_faces);
    do {
      for (const auto &edge_index : flippable_edge_indices_set) {
        auto &edge = this->get_checked_edge(edge_index);

        if (!this->is_edge_flippable_anisotropic_aware(edge)) {
          continue;
        }

        auto mesh_diff = this->flip_edge_triangulate(edge.get_self_index(), false);

        /* For each new edge added, set it's sizing */
        for (const auto &edge_index : mesh_diff.get_added_edges()) {
          auto &edge = this->get_checked_edge(edge_index);
          this->edge_set_size(edge);
        }

#if SHOULD_REMESH_DUMP_FILE
        auto after_flip_msgpack = this->serialize();
        auto after_flip_filename = static_remesh_name_gen.get_curr("after_flip");
        static_remesh_name_gen.gen_next();
        dump_file(after_flip_filename, after_flip_msgpack);
#endif

        /* Update `active_faces` */
        {
          /* Update `active_faces` to contain only face indices that
           * still exist in the mesh */
          blender::Vector<FaceIndex> new_active_faces;
          for (const auto &face_index : active_faces) {
            if (this->does_face_exist(face_index)) {
              new_active_faces.append(face_index);
            }
          }
          active_faces = std::move(new_active_faces);

          /* Add the newly created faces */
          active_faces.extend(mesh_diff.get_added_faces().as_span());
        }
      }

      flippable_edge_indices_set = this->get_flippable_edge_indices_set(active_faces);
      loop_cycles_until_now++;
    } while (flippable_edge_indices_set.size() != 0 && loop_cycles_until_now != max_loop_cycles);
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
        auto op_edge = this->get_edges().get(edge_index);
        if (!op_edge) {
          continue;
        }
        auto &edge = this->get_checked_edge(edge_index);
        auto mesh_diff = this->split_edge_triangulate(edge.get_self_index(), true);

#if SHOULD_REMESH_DUMP_FILE
        auto after_split_msgpack = this->serialize();
        auto after_split_filename = static_remesh_name_gen.get_curr("after_split");
        static_remesh_name_gen.gen_next();
        dump_file(after_split_filename, after_split_msgpack);
#endif

        /* For each new edge added, set it's sizing */
        for (const auto &edge_index : mesh_diff.get_added_edges()) {
          auto &edge = this->get_checked_edge(edge_index);
          this->edge_set_size(edge);
        }

        /* Flip edges of those faces that were created during the
         * split edge operation */
        auto added_faces = mesh_diff.get_added_faces();
        this->flip_edges(added_faces);
      }

      splittable_edges_set = this->get_splittable_edge_indices_set();
    } while (splittable_edges_set.size() != 0);
  }

  /**
   * Collapses edges whose "size" is less than (1.0 - small value)
   *
   * Based on [1]
   *
   * Here "size" is determined by `Sizing` stores in `Vert`s of the
   * `Edge`, using the function `Sizing::get_edge_size_sq()`.
   */
  void collapse_edges()
  {
    blender::Set<FaceIndex> active_faces;
    for (const auto &face : this->get_faces()) {
      active_faces.add_new(face.get_self_index());
    }

    do {
      /* It is not possible to iterate over active_faces and also
       * modify at the same time so store the new active faces in a
       * new set */
      blender::Set<FaceIndex> new_active_faces;

      for (const auto &face_index : active_faces) {
        const auto &op_face = this->get_faces().get(face_index);
        if (op_face == std::nullopt) {
          /* A previous edge collapse might have modified the this
           * face, so just continue onto the next one */
          continue;
        }
        const auto &face = this->get_checked_face(face_index);
        const auto edge_indices = this->get_edge_indices_of_face(face);

        for (const auto &edge_index : edge_indices) {
          const auto &op_edge = this->get_edges().get(edge_index);
          if (op_edge == std::nullopt) {
            break;
          }
          const auto &edge = this->get_checked_edge(edge_index);

          std::optional<AdaptiveMeshDiff<END>> op_mesh_diff = std::nullopt;
          if (this->is_edge_collapseable_adaptivemesh(edge, false)) {
            op_mesh_diff = this->collapse_edge_triangulate(edge.get_self_index(), false, true);
          }
          else if (this->is_edge_collapseable_adaptivemesh(edge, true)) {
            op_mesh_diff = this->collapse_edge_triangulate(edge.get_self_index(), true, true);
          }

          if (op_mesh_diff) {
#if SHOULD_REMESH_DUMP_FILE
            auto after_flip_msgpack = this->serialize();
            auto after_flip_filename = static_remesh_name_gen.get_curr("after_collapse");
            static_remesh_name_gen.gen_next();
            dump_file(after_flip_filename, after_flip_msgpack);
#endif
            const auto mesh_diff = op_mesh_diff.value();

            /* Must run flip edges on the newly added faces and
             * together the newly added faces must be added to
             * new_active_faces */
            {
              auto active_faces_from_flip_edges = mesh_diff.get_added_faces();
              this->flip_edges(active_faces_from_flip_edges);

              for (const auto &added_face : active_faces_from_flip_edges) {
                new_active_faces.add_new(added_face);
              }
            }
          }
        }
      }

      active_faces = std::move(new_active_faces);
    } while (active_faces.size() != 0);
  }

  void static_remesh(const Sizing &sizing)
  {
#if SHOULD_REMESH_DUMP_FILE
    auto static_remesh_start_msgpack = this->serialize();
    auto static_remesh_start_filename = static_remesh_name_gen.get_curr("static_remesh_start");
    static_remesh_name_gen.gen_next();
    dump_file(static_remesh_start_filename, static_remesh_start_msgpack);
#endif
    /* Set sizing for all verts */
    for (auto &vert : this->get_verts_mut()) {
      auto &op_vert_data = vert.get_extra_data_mut();
      if (op_vert_data) {
        auto &vert_data = op_vert_data.value();
        vert_data.set_sizing(sizing);
      }
      else {
        vert.set_extra_data(VertData(sizing));
      }
    }

    this->set_edge_sizes();

    /* Split the edges */
    this->split_edges();

    /* Collapse the edges */
    this->collapse_edges();

#if SHOULD_REMESH_DUMP_FILE
    auto static_remesh_end_msgpack = this->serialize();
    auto static_remesh_end_filename = static_remesh_name_gen.get_curr("static_remesh_end");
    static_remesh_name_gen.gen_next();
    dump_file(static_remesh_end_filename, static_remesh_end_msgpack);
#endif
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
    /* TODO(ish): Reference [1] says that the splittable edges should
     * be a set so this is done by checking if the verts are already
     * selected or not.
     *
     * This can lead to non symmetrical remeshing which wouldn't be
     * valid. So don't consider this at least for now. Will check
     * later again to see if it makes sense.
     *
     * An example of why selected verts may not work.
     *             v1__v2
     *              |  /
     *              | /
     *              |/
     *             /|v3
     *            / |
     *           /__|
     *          v5   v4
     *
     * Splitting (v1, v3) (v3, v4) can be done without it affecting
     * each other but one of the edges wouldn't be selected because v3
     * was already selected. This can lead to non symmetrical
     * splitting of the edges.
     */
    /* Deselect all verts */
    /* for (auto &vert : this->get_verts_mut()) { */
    /*   auto &vert_data = vert.get_checked_extra_data_mut(); */
    /*   auto &flag = vert_data.get_flag_mut(); */
    /*   flag &= ~VERT_SELECTED_FOR_SPLIT; */
    /* } */

    blender::Vector<EdgeIndex> splittable_edge_indices;
    /* It is assumed that the edges sizes have been computed earlier
     * and stored in the extra data of the edges */
    for (const auto &edge : this->get_edges()) {
      auto [v1, v2] = this->get_checked_verts_of_edge(edge, false);
      /* if (v1.get_checked_extra_data().get_flag() & VERT_SELECTED_FOR_SPLIT || */
      /*     v2.get_checked_extra_data().get_flag() & VERT_SELECTED_FOR_SPLIT) { */
      /*   continue; */
      /* } */
      const auto &edge_data = edge.get_checked_extra_data();
      auto edge_size = edge_data.get_size();
      if (edge_size > 1.0) {
        splittable_edge_indices.append(edge.get_self_index());
        auto &v1_data = v1.get_checked_extra_data_mut();
        auto &v1_flag = v1_data.get_flag_mut();
        v1_flag |= VERT_SELECTED_FOR_SPLIT;
        auto &v2_data = v2.get_checked_extra_data_mut();
        auto &v2_flag = v2_data.get_flag_mut();
        v2_flag |= VERT_SELECTED_FOR_SPLIT;
      }
    }

    /* Sort all the splittable edges based on their edge size,
     * largest to smallest. */
    std::sort(splittable_edge_indices.begin(),
              splittable_edge_indices.end(),
              [this](const auto &edge_index_1, const auto &edge_index_2) {
                const auto &edge_1 = this->get_checked_edge(edge_index_1);
                const auto &edge_2 = this->get_checked_edge(edge_index_2);

                return edge_1.get_checked_extra_data().get_size() >
                       edge_2.get_checked_extra_data().get_size();
              });

    return splittable_edge_indices;
  }

  /**
   * Checks if the edge is flippable nor not.
   *
   * Note: this is not the same as `Mesh::is_edge_flippable`, this is
   * specific to `AdaptiveMesh`.
   *
   * Reference [1] and [3]
   *
   * In this case considering [3] to be higher priority with a small change.
   */
  bool is_edge_flippable_anisotropic_aware(const AdaptiveEdge &edge) const
  {
    /* TODO(ish): expose alpha to the user */
    auto alpha = 0.1;

    if (this->is_edge_loose_or_on_seam_or_boundary(edge)) {
      return false;
    }
    if (this->is_edge_flippable(edge.get_self_index(), false) == false) {
      return false;
    }

    /* Flipping the edge should not cause the edge size metric to
     * fail.
     *
     * This condition is not part of reference [1] but it is important
     * so that the edges don't flip prematurely.
     */
    {
      const auto &ov1_index = this->get_checked_other_vert_index(edge.get_self_index(),
                                                                 edge.get_faces()[0]);
      const auto &ov2_index = this->get_checked_other_vert_index(edge.get_self_index(),
                                                                 edge.get_faces()[1]);

      const auto &ov1 = this->get_checked_vert(ov1_index);
      const auto &ov2 = this->get_checked_vert(ov2_index);

      if (this->compute_edge_size(ov1, ov2) > 1.0) {
        return false;
      }
    }

    const auto cross_2d = [](const float2 &a, const float2 &b) { return a.x * b.y - a.y * b.x; };

    /* Now the actual anisotropic aware critereon */
    /* Using the same convention as reference [1] */
    const auto [v_i_index, v_j_index] = edge.get_checked_verts();
    const auto v_k_index = this->get_checked_other_vert_index(edge.get_self_index(),
                                                              edge.get_faces()[0]);
    const auto v_l_index = this->get_checked_other_vert_index(edge.get_self_index(),
                                                              edge.get_faces()[1]);

    const auto &v_i = this->get_checked_vert(v_i_index);
    const auto &v_j = this->get_checked_vert(v_j_index);
    const auto &v_k = this->get_checked_vert(v_k_index);
    const auto &v_l = this->get_checked_vert(v_l_index);

    const auto &m_i = v_i.get_checked_extra_data().get_sizing();
    const auto &m_j = v_j.get_checked_extra_data().get_sizing();
    const auto &m_k = v_k.get_checked_extra_data().get_sizing();
    const auto &m_l = v_l.get_checked_extra_data().get_sizing();

    const auto u_jk = v_j.get_uv() - v_k.get_uv();
    const auto u_ik = v_i.get_uv() - v_k.get_uv();
    const auto u_il = v_i.get_uv() - v_l.get_uv();
    const auto u_jl = v_j.get_uv() - v_l.get_uv();

    const auto m_avg = (m_i + m_j + m_k + m_l) * 0.25;

    const auto lhs = cross_2d(u_jk, u_ik) * float2::dot(u_il, m_avg * u_jl) +
                     float2::dot(u_jk, m_avg * u_ik) * cross_2d(u_il, u_jl);

    /* Based on [1], should be flippable if res < 0.
     *
     * Based on [3], flippable if res falls some calculated value. So
     * taking that route as of now. But here, consider the absolute
     * value of the cross_2d values generated because they are just
     * calculating the area and the assumption is that the orientation
     * of the triangles shouldn't matter for this test.
     */
    const auto rhs = -alpha * (std::fabs(cross_2d(u_jk, u_ik)) + std::fabs(cross_2d(u_il, u_jl)));

    return lhs < rhs;
  }

  /**
   * Gets the maximal independent set of flippable edge indices in
   * `active_faces`.
   *
   * Reference [1]
   */
  blender::Vector<EdgeIndex> get_flippable_edge_indices_set(
      const blender::Vector<FaceIndex> &active_faces)
  {
    /* Deselect all verts of `active_faces` for flips */
    for (const auto &face_index : active_faces) {
      const auto &face = this->get_checked_face(face_index);

      for (const auto &vert_index : face.get_verts()) {
        auto &vert = this->get_checked_vert(vert_index);

        auto &vert_data = vert.get_checked_extra_data_mut();
        auto &flag = vert_data.get_flag_mut();
        flag &= ~VERT_SELECTED_FOR_FLIP;
      }
    }

    /* TODO(ish): Need to store a set of the edge indices and use
     * those because it is most likely that the `active_face` have
     * overlapping edges */

    blender::Vector<EdgeIndex> flippable_edge_indices;
    for (const auto &face_index : active_faces) {
      const auto &face = this->get_checked_face(face_index);

      const auto edge_indices = this->get_edge_indices_of_face(face);
      for (const auto &edge_index : edge_indices) {
        const auto &edge = this->get_checked_edge(edge_index);
        auto [v1, v2] = this->get_checked_verts_of_edge(edge, false);
        if (v1.get_checked_extra_data().get_flag() & VERT_SELECTED_FOR_FLIP ||
            v2.get_checked_extra_data().get_flag() & VERT_SELECTED_FOR_FLIP) {
          continue;
        }

        if (this->is_edge_flippable_anisotropic_aware(edge)) {
          flippable_edge_indices.append(edge.get_self_index());

          auto &v1_data = v1.get_checked_extra_data_mut();
          auto &v1_flag = v1_data.get_flag_mut();
          v1_flag |= VERT_SELECTED_FOR_FLIP;
          auto &v2_data = v2.get_checked_extra_data_mut();
          auto &v2_flag = v2_data.get_flag_mut();
          v2_flag |= VERT_SELECTED_FOR_FLIP;
        }
      }
    }

    return flippable_edge_indices;
  }

  bool is_edge_collapseable_adaptivemesh(const AdaptiveEdge &edge, bool verts_swapped) const
  {
    /* TODO(ish): expose small_value to gui */
    const auto small_value = 0.2;

    if (this->is_edge_collapseable(edge.get_self_index(), verts_swapped, true) == false) {
      return false;
    }

    const auto [v1, v2] = this->get_checked_verts_of_edge(edge, verts_swapped);

    /* If v1 is supposed to be preserved, cannot collapse the edge */
    {
      BLI_assert(v1.get_extra_data());
      if (v1.get_extra_data().value().get_flag() & VERT_PRESERVE) {
        return false;
      }
    }

    /* If v1 is on a seam or boundary, v2 should also be on a seam or boundary */
    if (this->is_vert_on_seam_or_boundary(v1) == true &&
        this->is_vert_on_seam_or_boundary(v2) == false) {
      /* This will modify the panel boundaries which isn't acceptable */
      return false;
    }

    /* Newly formed edges shouldn't exceed the edge size criterion */
    {
      const auto [v1_a, v2_a] = this->get_checked_verts_of_edge(edge, verts_swapped);
      const auto &n1_a = this->get_checked_node_of_vert(v1_a);
      const auto &n2_a = this->get_checked_node_of_vert(v2_a);
      const auto n1_index = n1_a.get_self_index();
      auto get_v1_v2_indices = [this, &n1_index, &verts_swapped](const AdaptiveEdge &e) {
        auto [v1, v2] = this->get_checked_verts_of_edge(e, verts_swapped);
        auto v1_index = v1.get_self_index();
        auto v2_index = v2.get_self_index();
        /* Need to swap the verts if v1 does not point to n1 */
        if (v1.get_node().value() != n1_index) {
          std::swap(v1_index, v2_index);
        }
        BLI_assert(this->get_checked_vert(v1_index).get_node().value() == n1_index);
        return std::make_tuple(v1_index, v2_index);
      };

      /* Get all 3D edges */
      const auto edge_indices = this->get_connecting_edge_indices(n1_a, n2_a);

      for (const auto &edge_index : edge_indices) {
        /* Get v1 of the 3D edge in correct order */
        const auto &e = this->get_checked_edge(edge_index);
        const auto [v1_index, v2_index] = get_v1_v2_indices(e);
        const auto &v1 = this->get_checked_vert(v1_index);
        const auto &v2 = this->get_checked_vert(v2_index);

        /* For edge adjacent to v1, check if the edge size is
         * exceeded if v1 is swapped for v2 */
        for (const auto &v1_edge_index : v1.get_edges()) {
          const auto &v1_edge = this->get_checked_edge(v1_edge_index);

          const auto v1_edge_verts = v1_edge.get_verts().value();

          const auto ov_index = std::get<0>(v1_edge_verts) == v1_index ?
                                    std::get<1>(v1_edge_verts) :
                                    std::get<0>(v1_edge_verts);

          const auto &ov = this->get_checked_vert(ov_index);
          const auto edge_size = this->compute_edge_size(v2, ov);

          if (edge_size > (1.0 - small_value)) {
            return false;
          }
        }
      }
    }

    /* TODO(ish): aspect ratio test */
    return true;
  }
};

}  // namespace blender::bke::internal

namespace msgpack {
MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{
  namespace adaptor {

  template<> struct pack<float[3]> {
    template<typename Stream>
    msgpack::packer<Stream> &operator()(msgpack::packer<Stream> &o, const float (&v)[3]) const
    {
      o.pack_array(3);

      o.pack(v[0]);
      o.pack(v[1]);
      o.pack(v[2]);

      return o;
    }
  };

  template<> struct pack<ClothVertex> {
    template<typename Stream>
    msgpack::packer<Stream> &operator()(msgpack::packer<Stream> &o, const ClothVertex &v) const
    {
      /* This check is to ensure that any new element added to
       * ClothVertex is also updated here. After adding the
       * interpolated value for the element (if needed), set the
       * correct sizeof(ClothVertex) in the assertion below. */
      BLI_assert(sizeof(ClothVertex) == 168);

      o.pack_array(22);

      o.pack(v.flags);
      o.pack(v.v);
      o.pack(v.xconst);
      o.pack(v.x);
      o.pack(v.xold);
      o.pack(v.tx);
      o.pack(v.txold);
      o.pack(v.tv);
      o.pack(v.mass);
      o.pack(v.goal);
      o.pack(v.impulse);
      o.pack(v.xrest);
      o.pack(v.dcvel);
      o.pack(v.impulse_count);
      o.pack(v.avg_spring_len);
      o.pack(v.struct_stiff);
      o.pack(v.bend_stiff);
      o.pack(v.shear_stiff);
      o.pack(v.spring_count);
      o.pack(v.shrink_factor);
      o.pack(v.internal_stiff);
      o.pack(v.pressure_factor);

      return o;
    }
  };

  template<typename T> struct pack<blender::bke::internal::NodeData<T>> {
    template<typename Stream>
    msgpack::packer<Stream> &operator()(msgpack::packer<Stream> &o,
                                        const blender::bke::internal::NodeData<T> &v) const
    {
      o.pack_array(2);

      o.pack(std::string("node_data"));
      o.pack(v.get_extra_data());

      return o;
    }
  };

  template<> struct pack<blender::bke::internal::ClothNodeData> {
    template<typename Stream>
    msgpack::packer<Stream> &operator()(msgpack::packer<Stream> &o,
                                        const blender::bke::internal::ClothNodeData &v) const
    {
      o.pack_array(2);

      o.pack(std::string("cloth_node_data"));
      o.pack(v.get_cloth_node_data());

      return o;
    }
  };

  template<> struct pack<blender::bke::internal::VertData> {
    template<typename Stream>
    msgpack::packer<Stream> &operator()(msgpack::packer<Stream> &o,
                                        const blender::bke::internal::VertData &v) const
    {
      o.pack_array(3);

      o.pack(std::string("vert_data"));
      o.pack(v.get_sizing());
      o.pack(v.get_flag());

      return o;
    }
  };

  template<> struct pack<blender::bke::internal::EdgeData> {
    template<typename Stream>
    msgpack::packer<Stream> &operator()(msgpack::packer<Stream> &o,
                                        const blender::bke::internal::EdgeData &v) const
    {
      o.pack_array(2);

      o.pack(std::string("edge_data"));
      o.pack(v.get_size());

      return o;
    }
  };

  template<> struct pack<blender::float2x2> {
    template<typename Stream>
    msgpack::packer<Stream> &operator()(msgpack::packer<Stream> &o,
                                        const blender::float2x2 &v) const
    {
      o.pack_array(4);

      o.pack(v.ptr()[0][0]);
      o.pack(v.ptr()[0][1]);
      o.pack(v.ptr()[1][0]);
      o.pack(v.ptr()[1][1]);

      return o;
    }
  };

  template<> struct pack<blender::bke::internal::Sizing> {
    template<typename Stream>
    msgpack::packer<Stream> &operator()(msgpack::packer<Stream> &o,
                                        const blender::bke::internal::Sizing &v) const
    {
      o.pack_array(2);

      o.pack(std::string("sizing"));
      o.pack(v.get_m());

      return o;
    }
  };

  }  // namespace adaptor
}  // MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
}  // namespace msgpack

namespace blender::bke {

template<typename END, typename ExtraData>
Mesh *adaptive_remesh(const AdaptiveRemeshParams<END, ExtraData> &params,
                      Mesh *mesh,
                      ExtraData &extra_data)
{
  internal::MeshIO meshio_input;
  meshio_input.read(mesh);

  internal::AdaptiveMesh<END> adaptive_mesh;
  adaptive_mesh.read(meshio_input);
  {
    const auto serialized = adaptive_mesh.serialize();
    internal::dump_file("/tmp/test.mesh", serialized);
  }

  /* Compute all the face normals, cannot rely on face normals set
   * when reading from the internal::MeshIO */
  adaptive_mesh.compute_face_normal_all_faces();

  /* Load up the `NodeData`'s extra_data */
  {
    auto i = 0;
    for (auto &node : adaptive_mesh.get_nodes_mut()) {
      node.set_extra_data(internal::NodeData(params.extra_data_to_end(extra_data, i)));
      i++;
    }

    params.post_extra_data_to_end(extra_data);
  }

  /* Important to not mess with the panel boundaries so if a vert is
   * marked for preserve it will not be removed and this takes care of
   * that. */
  adaptive_mesh.mark_verts_for_preserve();

  /* Actual Remeshing Part */
  {
    float size_min = params.size_min;
    auto m = float2x2::identity();
    m = m * (1.0 / size_min);
    internal::Sizing vert_sizing(std::move(m));
    adaptive_mesh.static_remesh(vert_sizing);
  }

  /* set back data from `AdaptiveMesh` to whatever needs it */
  {
    params.pre_end_to_extra_data(extra_data, adaptive_mesh.get_nodes().size());

    auto i = 0;
    for (const auto &node : adaptive_mesh.get_nodes()) {
      const auto &op_extra_data = node.get_extra_data();
      BLI_assert(op_extra_data);
      params.end_to_extra_data(extra_data, op_extra_data.value().get_extra_data(), i);
      i++;
    }
  }

  auto meshio_output = adaptive_mesh.write();
  return meshio_output.write();
}

/* This is needed because the compiler needs to know which
 * instantiations to make while compiling a templated function's .cpp
 * file
 *
 * reference:
 * https://isocpp.org/wiki/faq/templates#separate-template-fn-defn-from-decl
 */
template<>
Mesh *adaptive_remesh<internal::ClothNodeData, Cloth>(
    const AdaptiveRemeshParams<internal::ClothNodeData, Cloth> &, Mesh *, Cloth const &);

template<>
Mesh *adaptive_remesh<internal::EmptyExtraData, internal::EmptyExtraData>(
    AdaptiveRemeshParams<internal::EmptyExtraData, internal::EmptyExtraData> const &,
    Mesh *,
    internal::EmptyExtraData const &);

Mesh *BKE_cloth_remesh(Object *ob, ClothModifierData *clmd, Mesh *mesh)
{
  auto *cloth_to_object_res = cloth_to_object(ob, clmd, mesh, false);
  BLI_assert(cloth_to_object_res == nullptr);

  AdaptiveRemeshParams<internal::ClothNodeData, Cloth> params;
  params.size_min = clmd->sim_parms->remeshing_size_min;
  params.extra_data_to_end = [](const Cloth &cloth, size_t index) {
    BLI_assert(index < cloth.mvert_num);
    BLI_assert(cloth.verts);
    return internal::ClothNodeData(cloth.verts[index]);
  };
  params.post_extra_data_to_end = [](Cloth &cloth) {
    /* Delete the cloth.verts since it is stored within the `AdaptiveMesh` */
    BLI_assert(cloth.verts);
    MEM_freeN(cloth.verts);
    cloth.verts = nullptr;
    cloth.mvert_num = 0;
  };

  params.end_to_extra_data = [](Cloth &cloth, internal::ClothNodeData node_data, size_t index) {
    BLI_assert(index < cloth.mvert_num);
    BLI_assert(cloth.verts);
    cloth.verts[index] = node_data.get_cloth_node_data();
  };
  params.pre_end_to_extra_data = [](Cloth &cloth, size_t num_nodes) {
    /* caller should have deleted the verts earlier */
    BLI_assert(cloth.verts == nullptr);
    BLI_assert(cloth.mvert_num == 0);

    cloth.verts = static_cast<ClothVertex *>(
        MEM_callocN(sizeof(ClothVertex) * num_nodes, "cloth pre_end_to_extra_data"));
    BLI_assert(cloth.verts);

    cloth.mvert_num = num_nodes;
  };

  return adaptive_remesh(params, mesh, *clmd->clothObject);
}

Mesh *__temp_empty_adaptive_remesh(const TempEmptyAdaptiveRemeshParams &input_params, Mesh *mesh)
{
  using EmptyData = internal::EmptyExtraData;

  EmptyData empty_data;

  AdaptiveRemeshParams<EmptyData, EmptyData> params;
  params.size_min = input_params.size_min;
  params.extra_data_to_end = [](const EmptyData &UNUSED(data), size_t UNUSED(index)) {
    return internal::EmptyExtraData();
  };
  params.post_extra_data_to_end = [](EmptyData &UNUSED(cloth)) {};

  params.end_to_extra_data =
      [](EmptyData &UNUSED(data), EmptyData UNUSED(node_data), size_t UNUSED(index)) {};
  params.pre_end_to_extra_data = [](EmptyData &UNUSED(data), size_t UNUSED(num_nodes)) {};

  return adaptive_remesh(params, mesh, empty_data);
}

}  // namespace blender::bke
