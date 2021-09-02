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

#include "BLI_float2.hh"
#include "BLI_vector.hh"
#include "DNA_cloth_types.h"
#include "DNA_mesh_types.h"
#include "DNA_modifier_types.h"
#include "DNA_object_types.h"

#include "BLI_assert.h"
#include "BLI_float2x2.hh"
#include "BLI_kdopbvh.h"
#include "BLI_math.h"
#include "BLI_math_vector.h"
#include "BLI_utildefines.h"

#include "MEM_guardedalloc.h"

#include "BKE_cloth.h"
#include "BKE_cloth_remesh.hh"
#include "BKE_modifier.h"

#include "SIM_mass_spring.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>

#define SHOULD_REMESH_DUMP_FILE 1

namespace blender::bke::internal {
static FilenameGen remesh_name_gen("/tmp/remesh/remesh", ".mesh");

static std::string get_number_as_string(usize number)
{
  char number_str_c[16];
  BLI_snprintf(number_str_c, 16, "%05lu", number);
  std::string number_str(number_str_c);

  return number_str;
}

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
    /* CLOTH_VERT_FLAG_PINNED should not be propagated so that the
     * artist has the choice of pinning only verts that they want */
    cn.flags = (this->cloth_node_data.flags | other.cloth_node_data.flags) &
               (CLOTH_VERT_FLAG_NOSELFCOLL | CLOTH_VERT_FLAG_NOOBJCOLL);
    interp_v3_v3v3(cn.v, this->cloth_node_data.v, other.cloth_node_data.v, 0.5);
    interp_v3_v3v3(cn.xconst, this->cloth_node_data.xconst, other.cloth_node_data.xconst, 0.5);
    interp_v3_v3v3(cn.x, this->cloth_node_data.x, other.cloth_node_data.x, 0.5);
    interp_v3_v3v3(cn.xold, this->cloth_node_data.xold, other.cloth_node_data.xold, 0.5);
    interp_v3_v3v3(cn.tx, this->cloth_node_data.tx, other.cloth_node_data.tx, 0.5);
    interp_v3_v3v3(cn.txold, this->cloth_node_data.txold, other.cloth_node_data.txold, 0.5);
    interp_v3_v3v3(cn.tv, this->cloth_node_data.tv, other.cloth_node_data.tv, 0.5);
    cn.mass = simple_interp(this->cloth_node_data.mass, other.cloth_node_data.mass);
    /* No new nodes should have a goal since the artist has not added
     * the node and iterpolating the weights for newly added nodes can
     * lead to unexpected results that cannot be fixed by the
     * artist */
    cn.goal = 0.0;
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

enum EdgeFlags {
  EDGE_BETWEEN_SEWING_EDGES = 1 << 0,
};

class EdgeData {
  float size; /* from [1], the size calculated by the `Vert`s of the
               * `Edge` */
  /* EdgeFlags */
  uint32_t flags;

 public:
  EdgeData(float size) : size(size), flags(0)
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

  const auto &get_flags() const
  {
    return this->flags;
  }

  auto &get_flags_mut()
  {
    return this->flags;
  }

  EdgeData interp(const EdgeData &other) const
  {
    EdgeData res(std::numeric_limits<float>::signaling_NaN());

    res.flags = 0;
    /* Only if both the edge data are marked for
     * `EDGE_BETWEEN_SEWING_EDGES`, the result should be marked for
     * `EDGE_BETWEEN_SEWING_EDGES` */
    if ((this->flags & EDGE_BETWEEN_SEWING_EDGES) && (other.flags & EDGE_BETWEEN_SEWING_EDGES)) {
      res.flags |= EDGE_BETWEEN_SEWING_EDGES;
    }

    return res;
  }
};

class FaceData {
  float uv_area;

 public:
  FaceData(float uv_area) : uv_area(uv_area)
  {
  }

  const auto &get_uv_area() const
  {
    return this->uv_area;
  }

  void set_uv_area(float uv_area)
  {
    this->uv_area = uv_area;
  }
};

template<typename T> using AdaptiveNode = Node<NodeData<T>>;
using AdaptiveVert = Vert<VertData>;
using AdaptiveEdge = Edge<EdgeData>;
using AdaptiveFace = Face<FaceData>;
template<typename T> using AdaptiveMeshDiff = MeshDiff<NodeData<T>, VertData, EdgeData, FaceData>;

template<typename END>
class AdaptiveMesh : public Mesh<NodeData<END>, VertData, EdgeData, FaceData> {
 public:
  std::string serialize() const
  {
    std::stringstream ss;
    ss << "GenericAdaptiveMesh" << std::endl;

    msgpack::pack(ss, *this);

    return ss.str();
  }

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
   *
   * Important to not mess with the panel boundaries so if a vert is
   * marked for preserve it will not be removed and this takes care of
   * that.
   *
   * It also ensures that when sewing is enabled, all verts attached
   * to sewing edge(s) are also preserved.
   */
  void mark_verts_for_preserve(bool sewing_enabled)
  {
    for (auto &vert : this->get_verts_mut()) {
      if (this->is_vert_on_seam_or_boundary(vert)) {
        this->mark_vert_for_preserve(vert);
      }
    }

    /* Mark all verts attached to sewing edge(s) as preserve, this
     * ensures that no sewing edge(s) are removed which would
     * otherwise lead to results are not in line with what the artist
     * would want. */
    if (sewing_enabled) {
      for (const auto &edge : this->get_edges()) {
        if (edge.is_loose()) {
          const auto [v1_index, v2_index] = edge.get_checked_verts();
          auto &v1 = this->get_checked_vert(v1_index);
          this->mark_vert_for_preserve(v1);

          auto &v2 = this->get_checked_vert(v2_index);
          this->mark_vert_for_preserve(v2);
        }
      }
    }
  }

  /**
   * Flip edges of the `active_faces` if needed.
   *
   * Updates the active_faces in place
   */
  AdaptiveMeshDiff<END> flip_edges(blender::Vector<FaceIndex> &active_faces)
  {
    AdaptiveMeshDiff<END> complete_mesh_diff;
    auto max_loop_cycles = active_faces.size() * 3;
    auto loop_cycles_until_now = 0;
    auto flippable_edge_indices_set = this->get_flippable_edge_indices_set(active_faces);
    do {
      for (const auto &edge_index : flippable_edge_indices_set) {
        auto &edge = this->get_checked_edge(edge_index);

        if (!this->is_edge_flippable_anisotropic_aware(edge)) {
          continue;
        }

        const auto mesh_diff = this->flip_edge_triangulate(edge.get_self_index(), false);
        complete_mesh_diff.append(mesh_diff);

        this->compute_info_adaptivemesh(mesh_diff);

#if SHOULD_REMESH_DUMP_FILE
        auto after_flip_msgpack = this->serialize();
        auto after_flip_filename = remesh_name_gen.get_curr(
            "after_flip_" + get_number_as_string(std::get<0>(edge_index.get_raw())));
        remesh_name_gen.gen_next();
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

    /* Since `complete_mesh_diff` is not used for operations within
     * this function, `remove_non_existing_elements()` can be called
     * only once here before returning the `complete_mesh_diff` */
    complete_mesh_diff.remove_non_existing_elements(*this);
    return complete_mesh_diff;
  }

  /**
   * Splits edges whose "size" is greater than 1.0
   *
   * If sewing is enabled, it tried to add sewing edges if necessary.
   *
   * Based on [1]
   *
   * Here "size" is determined by `Sizing` stores in `Vert`s of the
   * `Edge`, using the function `Sizing::get_edge_size_sq()`.
   */
  void split_edges(bool sewing_enabled, bool force_split_for_sewing)
  {
    auto splittable_edges_set = this->get_splittable_edge_indices_set();
    do {
      for (const auto &edge_index : splittable_edges_set) {
        auto op_edge = this->get_edges().get(edge_index);
        if (!op_edge) {
          continue;
        }

        this->split_edge_adaptivemesh(edge_index, sewing_enabled, force_split_for_sewing);
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
            auto after_flip_filename = remesh_name_gen.get_curr(
                "after_collapse_" + get_number_as_string(std::get<0>(edge_index.get_raw())));
            remesh_name_gen.gen_next();
            dump_file(after_flip_filename, after_flip_msgpack);
#endif
            const auto mesh_diff = op_mesh_diff.value();

            this->compute_info_adaptivemesh(mesh_diff);

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

  template<typename ExtraData>
  void static_remesh(const Sizing &sizing, const AdaptiveRemeshParams<END, ExtraData> &params)
  {
#if SHOULD_REMESH_DUMP_FILE
    auto static_remesh_start_msgpack = this->serialize();
    auto static_remesh_start_filename = remesh_name_gen.get_curr("static_remesh_start");
    remesh_name_gen.gen_next();
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

    /* Set edge sizes */
    this->set_edge_sizes();

    bool sewing_enabled = params.flags & ADAPTIVE_REMESH_PARAMS_SEWING;
    bool force_split_for_sewing = params.flags & ADAPTIVE_REMESH_PARAMS_FORCE_SPLIT_FOR_SEWING;

    /* Mark edges that are between sewing edges only if sewing is
     * enabled */
    if (sewing_enabled) {
      this->mark_edges_between_sewing_edges();
    }

    /* Split the edges */
    this->split_edges(sewing_enabled, force_split_for_sewing);

    /* Collapse the edges */
    this->collapse_edges();

#if SHOULD_REMESH_DUMP_FILE
    auto static_remesh_end_msgpack = this->serialize();
    auto static_remesh_end_filename = remesh_name_gen.get_curr("static_remesh_end");
    remesh_name_gen.gen_next();
    dump_file(static_remesh_end_filename, static_remesh_end_msgpack);
#endif
  }

  template<typename ExtraData>
  void dynamic_remesh(const AdaptiveRemeshParams<END, ExtraData> &params)
  {
    /* TODO(ish): merge static_remesh and dynamic_remesh functions
     * since they differ only by a small part. Keeping it separate for
     * now for testing purposes. */
#if SHOULD_REMESH_DUMP_FILE
    auto dynamic_remesh_start_msgpack = this->serialize();
    auto dynamic_remesh_start_filename = remesh_name_gen.get_curr("dynamic_remesh_start");
    remesh_name_gen.gen_next();
    dump_file(dynamic_remesh_start_filename, dynamic_remesh_start_msgpack);
#endif

    /* TODO(ish): set vert sizing */

    /* Set edge sizes */
    this->set_edge_sizes();

    bool sewing_enabled = params.flags & ADAPTIVE_REMESH_PARAMS_SEWING;
    bool force_split_for_sewing = params.flags & ADAPTIVE_REMESH_PARAMS_FORCE_SPLIT_FOR_SEWING;

    /* Mark edges that are between sewing edges only if sewing is
     * enabled */
    if (sewing_enabled) {
      this->mark_edges_between_sewing_edges();
    }

    /* Split the edges */
    this->split_edges(sewing_enabled, force_split_for_sewing);

    /* Collapse the edges */
    this->collapse_edges();

#if SHOULD_REMESH_DUMP_FILE
    auto dynamic_remesh_end_msgpack = this->serialize();
    auto dynamic_remesh_end_filename = remesh_name_gen.get_curr("dynamic_remesh_end");
    remesh_name_gen.gen_next();
    dump_file(dynamic_remesh_end_filename, dynamic_remesh_end_msgpack);
#endif
  }

 private:
  bool is_edge_splittable_adaptivemesh(const AdaptiveEdge &edge) const
  {
    /* auto [v1, v2] = this->get_checked_verts_of_edge(edge, false); */
    /* if (v1.get_checked_extra_data().get_flag() & VERT_SELECTED_FOR_SPLIT || */
    /*     v2.get_checked_extra_data().get_flag() & VERT_SELECTED_FOR_SPLIT) { */
    /*   continue; */
    /* } */

    /* A loose edge should not be split */
    if (edge.is_loose()) {
      return false;
    }

    const auto &edge_data = edge.get_checked_extra_data();
    auto edge_size = edge_data.get_size();
    return edge_size > 1.0;
  }

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
      if (this->is_edge_splittable_adaptivemesh(edge)) {
        auto [v1, v2] = this->get_checked_verts_of_edge(edge, false);
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
   * Split the given edge and handle adaptivemesh specific
   * requirements like running `this->flip_edges` on faces created
   * during splitting, handling sewing if enabled.
   *
   * Returns a tuple of the MeshDiff of the entire split edge
   * operation (includes sewing related and flip edges operations) and
   * the set of verts that were added by the split operation only.
   */
  std::tuple<AdaptiveMeshDiff<END>, blender::Vector<VertIndex>> split_edge_adaptivemesh(
      const EdgeIndex &edge_index, bool sewing_enabled, bool force_split_for_sewing)
  {
    auto &edge = this->get_checked_edge(edge_index);
    auto mesh_diff = this->split_edge_triangulate(edge.get_self_index(), true, true);

#if SHOULD_REMESH_DUMP_FILE
    auto after_split_msgpack = this->serialize();
    auto after_split_filename = remesh_name_gen.get_curr(
        "after_split_" + get_number_as_string(std::get<0>(edge_index.get_raw())));
    remesh_name_gen.gen_next();
    dump_file(after_split_filename, after_split_msgpack);
#endif

    this->compute_info_adaptivemesh(mesh_diff);

    /* Store the verts added by the split edge operation to return
     * from the function */
    const auto added_verts = mesh_diff.get_added_verts();

    if (sewing_enabled) {
      BLI_assert(mesh_diff.get_added_nodes().size() == 1);
      const auto sewing_mesh_diff = this->try_adding_sewing_edge(mesh_diff.get_added_verts()[0],
                                                                 force_split_for_sewing);

      /* Append `sewing_mesh_diff` to `mesh_diff` so that
       * `flip_edges()` operates on a valid MeshDiff */
      mesh_diff.append(sewing_mesh_diff);
      mesh_diff.remove_non_existing_elements(*this);
    }

    /* Flip edges of those faces that were created during the
     * split edge operation */
    auto added_faces = mesh_diff.get_added_faces();
    const auto flip_edges_mesh_diff = this->flip_edges(added_faces);

    mesh_diff.append(flip_edges_mesh_diff);
    mesh_diff.remove_non_existing_elements(*this);

    return {mesh_diff, added_verts};
  }

  /**
   * Checks for the extra data flags of the edge to see if the edge is
   * flagged for `EDGE_BETWEEN_SEWING_EDGES`.
   *
   * Note: Caller must ensure edge has `extra_data`.
   */
  bool is_edge_between_sewing_edges(const AdaptiveEdge &edge) const
  {
    const auto &extra_data = edge.get_checked_extra_data();
    return extra_data.get_flags() & EDGE_BETWEEN_SEWING_EDGES;
  }

  /**
   * Tries to add a sewing edge to `vert_index` if it is possible.
   *
   * Adding a sewing edge is possible if the following checks are
   * true.
   *
   * => If there is a loose edge attached to one of the adjacent edges
   * of the given vert.
   *
   * => There is an adjacent edge to that loose edge which can be
   * split and it has another loose edge adjacent to it which can loop
   * back the given vert via an edge.
   */
  AdaptiveMeshDiff<END> try_adding_sewing_edge(const VertIndex &vert_index,
                                               bool force_split_for_sewing)
  {
    /* TODO(ish): make it work over 3D edges, so need to get Node
     * instead of Vert and then for each vert of the node, add the
     * sewing edge if needed */
    /* vert: is the vert that is being tested.
     *
     * e1: is an incident edge of `vert`.
     * e1_ov: is the other vertex of `e1` compared to `vert`.
     *
     * e2: is supposed to be the loose edge
     * e2_ov: is the other vertex of `e2` compared to `e1_ov`.
     *
     * opposite_edges: list of all edges that may be split to add the
     * sewing edge between `vert` and and the vert created when
     * `opposite_edge` is split.
     */
    /*
     *            e1   vert   e5
     *    e1_ov.________.________.e4_ov
     *         |                 |
     *       e2|                 |e4
     *         ._________________.
     *    e2_ov   opposite_edge   e3_ov
     *                 (e3)
     *
     */

    /* TODO(ish): Optimizations can be done, do an early check if
     * there are at least 2 edges incident on `vert` that have an
     * adjacent loose edge. (check if e1 and e5 exist).
     *
     * Another optimization can be to find the opposite edge before
     * splitting the edge. This does mean that it is not so
     * generalized, like it would not be able to add a sewing edge to
     * an existing vert but it will be a lot faster.
     */
    const auto &vert = this->get_checked_vert(vert_index);

    /* Need to ensure that the vert in question is in between 2 or
     * more edges that are between sewing edges. In case it is not, no
     * sewing edge from this vert should be created. */
    auto num_edges_between_sewing_edges = 0;
    for (const auto &e1_index : vert.get_edges()) {
      const auto &e1 = this->get_checked_edge(e1_index);

      if (this->is_edge_between_sewing_edges(e1)) {
        num_edges_between_sewing_edges++;
      }
    }
    if (num_edges_between_sewing_edges < 2) {
      return AdaptiveMeshDiff<END>();
    }

    blender::Vector<EdgeIndex> opposite_edges;

    /* Get the list of all opposite edges */
    for (const auto &e1_index : vert.get_edges()) {
      const auto &e1 = this->get_checked_edge(e1_index);

      const auto e1_ov_index = e1.get_checked_other_vert(vert_index);
      const auto &e1_ov = this->get_checked_vert(e1_ov_index);

      for (const auto &e2_index : e1_ov.get_edges()) {
        const auto &e2 = this->get_checked_edge(e2_index);

        if (e2.is_loose() == false) {
          continue;
        }

        const auto e2_ov_index = e2.get_checked_other_vert(e1_ov_index);
        const auto &e2_ov = this->get_checked_vert(e2_ov_index);

        for (const auto &e3_index : e2_ov.get_edges()) {
          const auto &e3 = this->get_checked_edge(e3_index);

          const auto e3_ov_index = e3.get_checked_other_vert(e2_ov_index);
          const auto &e3_ov = this->get_checked_vert(e3_ov_index);

          for (const auto &e4_index : e3_ov.get_edges()) {
            const auto &e4 = this->get_checked_edge(e4_index);

            if (e4.is_loose() == false) {
              continue;
            }

            const auto e4_ov_index = e4.get_checked_other_vert(e3_ov_index);
            const auto &e4_ov = this->get_checked_vert(e4_ov_index);

            for (const auto &e5_index : e4_ov.get_edges()) {
              const auto &e5 = this->get_checked_edge(e5_index);

              const auto e5_ov_index = e5.get_checked_other_vert(e4_ov_index);

              if (e5_ov_index == vert_index) {
                opposite_edges.append_non_duplicates(e3_index);
              }
            }
          }
        }
      }
    }

    /* Iterate over `opposite_edges`, if an `opposite_edge` is
     * splittable, then split it and create a new edge between the
     * `new_vert` and `vert` */

    AdaptiveMeshDiff<END> complete_mesh_diff;
    for (const auto &opposite_edge_index : opposite_edges) {
      /* It is possible that that splitting a previous `opposite_edge`
       * might have removed this edge */
      if (this->has_edge(opposite_edge_index) == false) {
        continue;
      }

      {
        const auto &opposite_edge = this->get_checked_edge(opposite_edge_index);

        if (force_split_for_sewing == false) {
          if (this->is_edge_splittable_adaptivemesh(opposite_edge) == false) {
            continue;
          }
        }

        if (this->is_edge_between_sewing_edges(opposite_edge) == false) {
          continue;
        }
      }

      const auto [mesh_diff, added_verts] = this->split_edge_adaptivemesh(
          opposite_edge_index, true, force_split_for_sewing);
      complete_mesh_diff.append(mesh_diff);
      complete_mesh_diff.remove_non_existing_elements(*this);

      /* TODO(ish): update this when the sewing edge is added between
       * nodes and not just verts. */
      /* Create an edge between given `vert` and `added_verts[0]` */
      BLI_assert(added_verts.size() >= 1);
      auto &new_edge = this->add_checked_loose_edge(vert_index, added_verts[0]);
      this->compute_info_edge_adaptivemesh(new_edge);

#if SHOULD_REMESH_DUMP_FILE
      auto after_adding_loose_edge_msgpack = this->serialize();
      auto after_adding_loose_edge_filename = remesh_name_gen.get_curr(
          "after_adding_loose_edge_" + get_number_as_string(std::get<0>(vert_index.get_raw())) +
          "_" + get_number_as_string(std::get<0>(added_verts[0].get_raw())));
      remesh_name_gen.gen_next();
      dump_file(after_adding_loose_edge_filename, after_adding_loose_edge_msgpack);
#endif

      /* Mark the new (loose) edge's verts as preserve */
      auto [new_edge_v1, new_edge_v2] = this->get_checked_verts_of_edge(new_edge, false);
      this->mark_vert_for_preserve(new_edge_v1);
      this->mark_vert_for_preserve(new_edge_v2);

      complete_mesh_diff.add_edge(new_edge.get_self_index());
    }

    return complete_mesh_diff;
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

    /* Should not flip an edge that creates face(s) that are inverted */
    {
      const auto &f1 = this->get_checked_face(edge.get_faces()[0]);
      const auto &f2 = this->get_checked_face(edge.get_faces()[1]);
      const auto &ov1 = this->get_checked_other_vert(edge, f1);
      const auto &ov2 = this->get_checked_other_vert(edge, f2);

      const auto [v1, v2] = this->get_checked_verts_of_edge(edge, false);

      const auto n1 = this->compute_face_normal(v1, ov2, ov1);
      const auto n2 = this->compute_face_normal(v2, ov1, ov2);
      const auto expected_normal = f1.get_normal() + f2.get_normal();

      if (float3::dot(n1, expected_normal) <= 0.0) {
        return false;
      }
      if (float3::dot(n2, expected_normal) <= 0.0) {
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

  /**
   * Calculates the aspect ratio of the triangle formed by the given
   * positions (p1, p2, p3).
   *
   * Refer to [4] for more information about the aspect ratio
   * calculation. Using "The measure associated with interpolation
   * error in the usual (weaker) bounds given by approximation
   * theory. Nonsmooth." in table 7.
   */
  float compute_aspect_ratio(const float2 &p1, const float2 &p2, const float2 &p3) const
  {
    /* It is possible to use "The aspect ratio, or ratio
     * between the minimum and maximum dimensions of the triangle.
     * Nonsmooth." from [4] but the current implemention seems to
     * provide results closer to that of [1] */

    const auto l1 = (p2 - p1).length();
    const auto l2 = (p3 - p2).length();
    const auto l3 = (p1 - p3).length();

    const auto l_max = max_ff(max_ff(l1, l2), l3);

    const auto cross_2d = [](const float2 &a, const float2 &b) { return a.x * b.y - a.y * b.x; };

    const auto area = fabs(cross_2d(p2 - p1, p3 - p1)) * 0.5;

    return 4.0 * M_SQRT3 * area / (l_max * (l1 + l2 + l3));
  }

  /**
   * Calculates the aspect ratio of the triangle described by the UVs
   * of the `Vert`s provided.
   */
  float compute_aspect_ratio(const AdaptiveVert &v1,
                             const AdaptiveVert &v2,
                             const AdaptiveVert &v3) const
  {
    return this->compute_aspect_ratio(v1.get_uv(), v2.get_uv(), v3.get_uv());
  }

  /**
   * Easy call to above when only indices available
   */
  float compute_aspect_ratio(const VertIndex &v1_index,
                             const VertIndex &v2_index,
                             const VertIndex &v3_index) const
  {
    return this->compute_aspect_ratio(this->get_checked_vert(v1_index),
                                      this->get_checked_vert(v2_index),
                                      this->get_checked_vert(v3_index));
  }

  /**
   * Calculates the aspect ratio of given triangle
   *
   * Note: This function asserts in debug mode that the given face is
   * a triangle. In release mode it will lead to undefined behaviour
   * when the number of verts in the face is not 3.
   */
  float compute_aspect_ratio(const AdaptiveFace &face) const
  {
    BLI_assert(face.get_verts().size() == 3);
    return this->compute_aspect_ratio(
        face.get_verts()[0], face.get_verts()[1], face.get_verts()[2]);
  }

  bool is_edge_collapseable_adaptivemesh(const AdaptiveEdge &edge, bool verts_swapped) const
  {
    /* TODO(ish): expose small_value, aspect_ratio_min to gui */
    const auto small_value = 0.2;
    const auto aspect_ratio_min = 0.1;

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

    /* Newly formed edges shouldn't exceed the edge size criterion
     * and newly formed faces shouldn't be inverted */
    {
      const auto [v1_a, v2_a] = this->get_checked_verts_of_edge(edge, verts_swapped);
      const auto v1_index = v1_a.get_self_index();
      const auto v2_index = v2_a.get_self_index();
      const auto &n1_a = this->get_checked_node_of_vert(v1_a);
      const auto &n2_a = this->get_checked_node_of_vert(v2_a);
      const auto n1_index = n1_a.get_self_index();

      /* Get all 3D edges */
      const auto edge_indices = this->get_connecting_edge_indices(n1_a, n2_a);

      for (const auto &edge_index : edge_indices) {
        /* Get v1 of the 3D edge in correct order */
        const auto &e = this->get_checked_edge(edge_index);
        const auto [v1_index, v2_index] = this->get_checked_vert_indices_of_edge_aligned_with_n1(
            e, n1_index);
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

      /* Face inversion check and aspect ratio check */
      const auto v1_face_indices = this->get_checked_face_indices_of_vert(v1_index);
      for (const auto &face_index : v1_face_indices) {
        auto &f = this->get_checked_face(face_index);

        /* Cannot create face between v2, v2, ov */
        if (f.has_vert_index(v2_index)) {
          continue;
        }

        BLI_assert(f.get_verts().size() == 3);

        blender::Array<VertIndex> vert_indices(f.get_verts().as_span());

        bool v2_exists = false;
        for (auto &vert_index : vert_indices) {
          if (vert_index == v2_index) {
            v2_exists = true;
            break;
          }
          if (vert_index == v1_index) {
            vert_index = v2_index;
            break;
          }
        }

        if (v2_exists) {
          continue;
        }

        const auto new_normal = this->compute_face_normal(this->get_checked_vert(vert_indices[0]),
                                                          this->get_checked_vert(vert_indices[1]),
                                                          this->get_checked_vert(vert_indices[2]));
        const auto &expected_normal = f.get_normal();

        /* Face inversion check */
        if (float3::dot(new_normal, expected_normal) <= 0.0) {
          return false;
        }

        /* Aspect ratio check */
        if (this->compute_aspect_ratio(vert_indices[0], vert_indices[1], vert_indices[2]) <
            aspect_ratio_min) {
          return false;
        }
      }
    }

    return true;
  }

  /**
   * Computes extra data information for given node
   */
  void compute_info_node_adaptivemesh(AdaptiveNode<END> &node)
  {
    this->compute_info_node(node);
  }

  /**
   * Computes extra data information for given vert
   */
  void compute_info_vert_adaptivemesh(AdaptiveVert &vert)
  {
    this->compute_info_vert(vert);
  }

  /**
   * Computes extra data information for given edge
   */
  void compute_info_edge_adaptivemesh(AdaptiveEdge &edge)
  {
    this->compute_info_edge(edge);

    /* For each new edge added, set it's sizing */
    this->edge_set_size(edge);
  }

  /**
   * Computes extra data information for given face
   */
  void compute_info_face_adaptivemesh(AdaptiveFace &face)
  {
    this->compute_info_face(face);
  }

  /**
   * Compute extra information for all the elements added (stored
   * within mesh_diff) */
  void compute_info_adaptivemesh(
      const MeshDiff<NodeData<END>, VertData, EdgeData, FaceData> &mesh_diff)
  {
    for (const auto &node_index : mesh_diff.get_added_nodes()) {
      auto &node = this->get_checked_node(node_index);
      this->compute_info_node_adaptivemesh(node);
    }

    for (const auto &vert_index : mesh_diff.get_added_verts()) {
      auto &vert = this->get_checked_vert(vert_index);
      this->compute_info_vert_adaptivemesh(vert);
    }

    for (const auto &edge_index : mesh_diff.get_added_edges()) {
      auto &edge = this->get_checked_edge(edge_index);
      this->compute_info_edge_adaptivemesh(edge);
    }

    for (const auto &face_index : mesh_diff.get_added_faces()) {
      auto &face = this->get_checked_face(face_index);
      this->compute_info_face_adaptivemesh(face);
    }
  }

  /**
   * For the given vert, sets the `VERT_PRESERVE` flag.
   */
  void mark_vert_for_preserve(AdaptiveVert &vert)
  {
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

  /**
   * Marks all edges that are between two sewing edges that are
   * connected by another edge as well.
   */
  void mark_edges_between_sewing_edges()
  {
    /*                    edge
     *        v1 ._____________________. v2
     *           |                     |
     *      v1_e |                     | v2_e
     *           |                     |
     *   v1_e_ov ._____________________. v2_e_ov
     *               connecting_edge
     *
     *
     * Here `v1_e` and `v2_e` must be loose and there must be a
     * connecting edge between `v1_e_ov` and `v2_e_ov` for the `edge`
     * to be marked as `EDGE_BETWEEN_SEWING_EDGES`.
     */
    for (auto &edge : this->get_edges_mut()) {
      const auto [v1_index, v2_index] = edge.get_checked_verts();

      const auto &v1 = this->get_checked_vert(v1_index);
      const auto &v2 = this->get_checked_vert(v2_index);

      blender::Vector<EdgeIndex> v1_loose_edges;
      for (const auto &v1_e_index : v1.get_edges()) {
        const auto &v1_e = this->get_checked_edge(v1_e_index);
        if (v1_e.is_loose()) {
          v1_loose_edges.append(v1_e_index);
        }
      }

      blender::Vector<EdgeIndex> v2_loose_edges;
      for (const auto &v2_e_index : v2.get_edges()) {
        const auto &v2_e = this->get_checked_edge(v2_e_index);
        if (v2_e.is_loose()) {
          v2_loose_edges.append(v2_e_index);
        }
      }

      for (const auto &v1_e_index : v1_loose_edges) {
        const auto &v1_e = this->get_checked_edge(v1_e_index);
        const auto &v1_e_ov_index = v1_e.get_checked_other_vert(v1_index);
        for (const auto &v2_e_index : v2_loose_edges) {
          const auto &v2_e = this->get_checked_edge(v2_e_index);
          const auto &v2_e_ov_index = v2_e.get_checked_other_vert(v2_index);

          if (this->get_connecting_edge_index(v1_e_ov_index, v2_e_ov_index)) {
            /* Mark `edge` as an edge between sewing edges */
            /* `edge`'s extra data should have been set by this point */
            auto &edge_data = edge.get_checked_extra_data_mut();
            auto &flags = edge_data.get_flags_mut();
            flags |= EDGE_BETWEEN_SEWING_EDGES;
          }
        }
      }
    }
  }
};

template<typename END> using AdaptiveNode = Node<NodeData<END>>;
using AdaptiveVert = Vert<VertData>;
using AdaptiveEdge = Edge<EdgeData>;
using AdaptiveFace = Face<FaceData>;
using EmptyAdaptiveMesh = AdaptiveMesh<EmptyExtraData>;
using ClothAdaptiveMesh = AdaptiveMesh<ClothNodeData>;

template<> std::string EmptyAdaptiveMesh::serialize() const
{
  std::stringstream ss;
  ss << "EmptyAdaptiveMesh" << std::endl;

  msgpack::pack(ss, *this);

  return ss.str();
}

template<> std::string ClothAdaptiveMesh::serialize() const
{
  std::stringstream ss;
  ss << "ClothAdaptiveMesh" << std::endl;

  msgpack::pack(ss, *this);

  return ss.str();
}

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
      o.pack_array(3);

      o.pack(std::string("edge_data"));
      o.pack(v.get_size());
      o.pack(v.get_flags());

      return o;
    }
  };

  template<> struct pack<blender::bke::internal::FaceData> {
    template<typename Stream>
    msgpack::packer<Stream> &operator()(msgpack::packer<Stream> &o,
                                        const blender::bke::internal::FaceData &v) const
    {
      o.pack_array(2);

      o.pack(std::string("face_data"));
      o.pack(v.get_uv_area());

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

  template<> struct pack<blender::bke::internal::EmptyAdaptiveMesh> {
    template<typename Stream>
    msgpack::packer<Stream> &operator()(
        msgpack::packer<Stream> &o, const blender::bke::internal::EmptyAdaptiveMesh &mesh) const
    {
      pack_mesh(o, mesh);

      return o;
    }
  };

  template<> struct pack<blender::bke::internal::ClothAdaptiveMesh> {
    template<typename Stream>
    msgpack::packer<Stream> &operator()(
        msgpack::packer<Stream> &o, const blender::bke::internal::ClothAdaptiveMesh &mesh) const
    {
      pack_mesh(o, mesh);

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
  bool sewing_enabled = params.flags & ADAPTIVE_REMESH_PARAMS_SEWING;
  adaptive_mesh.mark_verts_for_preserve(sewing_enabled);

  /* Actual Remeshing Part */
  if (params.type == ADAPTIVE_REMESH_PARAMS_STATIC_REMESH) {
    float size_min = params.size_min;
    auto m = float2x2::identity();
    m = m * (1.0 / size_min);
    internal::Sizing vert_sizing(std::move(m));
    adaptive_mesh.static_remesh(vert_sizing, params);
  }
  else if (params.type == ADAPTIVE_REMESH_PARAMS_DYNAMIC_REMESH) {
    adaptive_mesh.dynamic_remesh(params);
  }
  else {
    BLI_assert_unreachable();
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

/**
 * If the mesh has been updated, the caller must ensure that
 * clmd->clothObject->verts has been set and then call this to update
 * the rest of the cloth modifier to reflect the changes in the mesh.
 */
static void set_cloth_information_when_new_mesh(Object *ob, ClothModifierData *clmd, Mesh *mesh)
{
  BLI_assert(clmd != nullptr);
  BLI_assert(mesh != nullptr);
  BLI_assert(clmd->clothObject->verts != nullptr);
  BLI_assert(clmd->clothObject->mvert_num == mesh->totvert);

  cloth_from_mesh(clmd, ob, mesh, false);

  /* Build the springs */
  if (!cloth_build_springs(clmd, mesh)) {
    cloth_free_modifier(clmd);
    BKE_modifier_set_error(ob, &(clmd->modifier), "Cannot build springs");
    /* TODO(ish): error handling */
    /* return false; */
  }

  /* Set up the cloth solver */
  SIM_cloth_solver_free(clmd);
  SIM_cloth_solver_init(ob, clmd);
  SIM_cloth_solver_set_positions(clmd);

  /* Free any existing bvh trees */
  if (clmd->clothObject->bvhtree) {
    BLI_bvhtree_free(clmd->clothObject->bvhtree);
  }
  if (clmd->clothObject->bvhselftree) {
    BLI_bvhtree_free(clmd->clothObject->bvhselftree);
  }

  clmd->clothObject->bvhtree = bvhtree_build_from_cloth(clmd, clmd->coll_parms->epsilon);
  clmd->clothObject->bvhselftree = bvhtree_build_from_cloth(clmd, clmd->coll_parms->selfepsilon);
}

void BKE_cloth_serialize_adaptive_mesh(Object *ob,
                                       ClothModifierData *clmd,
                                       Mesh *mesh,
                                       const char *location)
{
  AdaptiveRemeshParams<internal::ClothNodeData, Cloth> params;
  params.size_min = clmd->sim_parms->remeshing_size_min;
  params.extra_data_to_end = [](const Cloth &cloth, size_t index) {
    BLI_assert(index < cloth.mvert_num);
    BLI_assert(cloth.verts);
    return internal::ClothNodeData(cloth.verts[index]);
  };
  params.post_extra_data_to_end = [](Cloth &UNUSED(cloth)) {
    /* Do nothing */
  };

  params.end_to_extra_data =
      [](Cloth &UNUSED(cloth), internal::ClothNodeData UNUSED(node_data), size_t UNUSED(index)) {
        /* Do nothing */
      };
#ifndef NDEBUG
  params.pre_end_to_extra_data = [](Cloth &cloth, size_t num_nodes) {
    /* Do not allocate cloth.verts, it shouldn't have been modified */
    BLI_assert(cloth.verts != nullptr);
    BLI_assert(cloth.mvert_num == num_nodes);
  };
#else
  params.pre_end_to_extra_data = [](Cloth &UNUSED(cloth), size_t UNUSED(num_nodes)) {
    /* Do nothing */
  };
#endif

  const auto remeshing = clmd->sim_parms->flags & CLOTH_SIMSETTINGS_FLAG_REMESH;
  Mesh *cloth_to_object_res = nullptr;
  if (remeshing && clmd->prev_frame_mesh) {
    cloth_to_object_res = cloth_to_object(ob, clmd, clmd->prev_frame_mesh, true);
  }
  else {
    cloth_to_object_res = cloth_to_object(ob, clmd, mesh, true);
  }

  internal::MeshIO meshio_input;
  meshio_input.read(cloth_to_object_res);

  internal::AdaptiveMesh<internal::ClothNodeData> adaptive_mesh;
  adaptive_mesh.read(meshio_input);

  Cloth &extra_data = *clmd->clothObject;

  /* Load up the `NodeData`'s extra_data */
  {
    auto i = 0;
    for (auto &node : adaptive_mesh.get_nodes_mut()) {
      node.set_extra_data(internal::NodeData(params.extra_data_to_end(extra_data, i)));
      i++;
    }

    params.post_extra_data_to_end(extra_data);
  }

  const auto serialized = adaptive_mesh.serialize();
  internal::dump_file(location, serialized);

  BKE_mesh_eval_delete(cloth_to_object_res);
}

Mesh *BKE_cloth_remesh(Object *ob, ClothModifierData *clmd, Mesh *mesh)
{
  if (clmd->prev_frame_mesh) {
    mesh = clmd->prev_frame_mesh;
  }

#ifndef NDEBUG
  Mesh *cloth_to_object_res = cloth_to_object(ob, clmd, mesh, false);
  BLI_assert(cloth_to_object_res == nullptr);
#else
  cloth_to_object(ob, clmd, mesh, false);
#endif

  AdaptiveRemeshParams<internal::ClothNodeData, Cloth> params;
  params.size_min = clmd->sim_parms->remeshing_size_min;
  params.flags = 0;
  if (clmd->sim_parms->flags & CLOTH_SIMSETTINGS_FLAG_SEW) {
    params.flags |= ADAPTIVE_REMESH_PARAMS_SEWING;
  }
  if (clmd->sim_parms->remeshing_type == CLOTH_REMESHING_STATIC) {
    params.type = ADAPTIVE_REMESH_PARAMS_STATIC_REMESH;
  }
  else if (clmd->sim_parms->remeshing_type == CLOTH_REMESHING_DYNAMIC) {
    params.type = ADAPTIVE_REMESH_PARAMS_DYNAMIC_REMESH;
  }
  else {
    BLI_assert_unreachable();
  }

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

  Mesh *new_mesh = adaptive_remesh(params, mesh, *clmd->clothObject);

  set_cloth_information_when_new_mesh(ob, clmd, new_mesh);

  if (clmd->prev_frame_mesh) {
    BKE_mesh_eval_delete(clmd->prev_frame_mesh);
    clmd->prev_frame_mesh = nullptr;
  }

  clmd->prev_frame_mesh = BKE_mesh_copy_for_eval(new_mesh, false);

  return new_mesh;
}

Mesh *__temp_empty_adaptive_remesh(const TempEmptyAdaptiveRemeshParams &input_params, Mesh *mesh)
{
  using EmptyData = internal::EmptyExtraData;

  EmptyData empty_data;

  AdaptiveRemeshParams<EmptyData, EmptyData> params;
  params.size_min = input_params.size_min;
  params.flags = input_params.flags;
  params.type = input_params.type;
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
