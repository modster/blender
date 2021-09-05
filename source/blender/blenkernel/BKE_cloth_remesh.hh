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
 * The Original Code is Copyright (C) Blender Foundation.
 * All rights reserved.
 */

#pragma once

/** \file
 * \ingroup bke
 */

/*********************************************************************************
 * references
 *
 * [1] "Adaptive Anisotropic Remeshing for Cloth Simulation" by Rahul
 * Narain, Armin Samii and James F. O'Brien (SIGGRAPH 2012)
 * http://graphics.berkeley.edu/papers/Narain-AAR-2012-11/index.html
 *
 * [2] "Adjacency and incidence framework: a data structure for
 * efficient and fast management of multiresolution meshes" by
 * Frutuoso G. M. Silva and Abel J. P. Gomes (GRAPHITE '03)
 * https://doi.org/10.1145/604471.604503
 *
 * [3] "Folding and Crumpling Adaptive Sheets" by Rahul Narain, Tobias
 * Pfaff, James F.O'Brien (SIGGRAPH 2013).
 * https://dl.acm.org/doi/10.1145/2461912.2462010
 * http://graphics.berkeley.edu/papers/Narain-FCA-2013-07/Narain-FCA-2013-07.pdf
 *
 * [4] "What Is a Good Linear Finite Element? Interpolation,
 * Conditioning, Anisotropy, and Quality Measures" by Jonathan Richard Shewchuk
 *
 * *****************************************************************************/

#include "BKE_mesh.h"
#include "BLI_assert.h"

#include "BKE_customdata.h"

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"

struct ClothModifierData;
struct Object;

#ifdef __cplusplus
namespace blender::bke {
extern "C" {
#endif

Mesh *BKE_cloth_remesh(struct Object *ob, struct ClothModifierData *clmd, struct Mesh *mesh);

void BKE_cloth_serialize_adaptive_mesh(struct Object *ob,
                                       struct ClothModifierData *clmd,
                                       struct Mesh *mesh,
                                       const char *location);

#ifdef __cplusplus
}
} /* namespace blender::bke */
#endif

#ifdef __cplusplus

#  include <bits/stdint-uintn.h>
#  include <cmath>
#  include <filesystem>
#  include <fstream>
#  include <functional>
#  include <iostream>
#  include <istream>
#  include <limits>
#  include <optional>
#  include <sstream>
#  include <string>
#  include <tuple>

#  include "msgpack.hpp"
#  include "msgpack/adaptor/define_decl.hpp"

#  include "BLI_array.hh"
#  include "BLI_float2.hh"
#  include "BLI_float2_msgpack.hh"
#  include "BLI_float3.hh"
#  include "BLI_float3_msgpack.hh"
#  include "BLI_generational_arena.hh"
#  include "BLI_generational_arena_msgpack.hh"
#  include "BLI_map.hh"
#  include "BLI_map_msgpack.hh"
#  include "BLI_set.hh"
#  include "BLI_vector.hh"
#  include "BLI_vector_msgpack.hh"

/* Public C++ code */
namespace blender::bke {

/**
 * @param END Extra Node Data
 *
 * @param ExtraData Extra Data that might be needed to get the `Extra
 * "Element" Data`
 */
template<typename END, typename ExtraData> struct AdaptiveRemeshParams {
  float edge_length_min;
  float edge_length_max;
  float aspect_ratio_min;
  float change_in_vertex_normal_max;
  /* AdaptiveRemeshParamsFlags */
  uint32_t flags;
  /* AdaptiveRemeshParamsType */
  uint32_t type;

  /* For handling Extra Node Data */
  /**
   * function that takes `ExtraData` along with the index for the
   * `Node` as input and returns the `END` to be stored in that
   * `Node`.
   */
  std::function<END(const ExtraData &, size_t)> extra_data_to_end;
  /**
   * function that is run after `extra_data_to_end` for every `Node` of
   * the `Mesh`.
   *
   * useful for cleanup. (memory management of resources within `ExtraData`)
   */
  std::function<void(ExtraData &)> post_extra_data_to_end;
  /**
   * function that takes `ExtraData` along with the `END` and
   * corresponding index of the `Node`
   *
   * useful to store `END` into `ExtraData`
   */
  std::function<void(ExtraData &, END, size_t)> end_to_extra_data;
  /**
   * function that takes `ExtraData` along with the number of `Node`s
   * in `Mesh`.
   *
   * is run before `end_to_extra_data` is run for every `Node` of the
   * `Mesh`.
   *
   * useful for memory management of resources within `ExtraData`.
   */
  std::function<void(ExtraData &, size_t)> pre_end_to_extra_data;
};

/**
 * AdaptiveRemeshParams->flags
 */
enum AdaptiveRemeshParamsFlags {
  /** Sewing is enabled */
  ADAPTIVE_REMESH_PARAMS_SEWING = 1 << 0,
  ADAPTIVE_REMESH_PARAMS_FORCE_SPLIT_FOR_SEWING = 1 << 1,
};

/**
 * AdaptiveRemeshParams->type
 */
enum AdaptiveRemeshParamsType {
  ADAPTIVE_REMESH_PARAMS_STATIC_REMESH = 0,
  ADAPTIVE_REMESH_PARAMS_DYNAMIC_REMESH = 1,
};

/* `mesh` cannot be made const because function defined on `struct
 * Mesh` do not take `struct Mesh` as const even when they can be const */
template<typename END, typename ExtraData>
Mesh *adaptive_remesh(const AdaptiveRemeshParams<END, ExtraData> &params,
                      Mesh *mesh,
                      const ExtraData &extra_data);

} /* namespace blender::bke */

namespace blender::bke::internal {

template<typename> class Node;
template<typename> class Vert;
template<typename> class Edge;
template<typename> class Face;
template<typename, typename, typename, typename> class Mesh;
class MeshIO;
template<typename, typename, typename, typename> class MeshDiff;
class EmptyExtraData {
 public:
  EmptyExtraData interp(const EmptyExtraData &other) const
  {
    return other;
  }
};

namespace ga = blender::generational_arena;
namespace fs = std::filesystem;

using NodeIndex = ga::Index;
using VertIndex = ga::Index;
using EdgeIndex = ga::Index;
using FaceIndex = ga::Index;
using IncidentVerts = blender::Vector<VertIndex>;
using IncidentEdges = blender::Vector<EdgeIndex>;
using IncidentFaces = blender::Vector<FaceIndex>;
using AdjacentVerts = IncidentVerts;
using EdgeVerts = std::tuple<VertIndex, VertIndex>;

using usize = uint64_t;

class FilenameGen {
  usize number;
  std::string prefix;
  std::string suffix;

 public:
  FilenameGen(const std::string prefix, const std::string suffix)
      : number(0), prefix(prefix), suffix(suffix)
  {
  }

  std::string gen_next()
  {
    this->number += 1;
    return this->get_curr();
  }

  std::string gen_next(const std::string pre_suffix)
  {
    this->number += 1;
    return this->get_curr(pre_suffix);
  }

  std::string get_curr()
  {
    char number_str_c[16];
    BLI_snprintf(number_str_c, 16, "%05lu", this->number);
    std::string number_str(number_str_c);
    return this->prefix + "_" + number_str + this->suffix;
  }

  std::string get_curr(const std::string pre_suffix)
  {
    char number_str_c[16];
    BLI_snprintf(number_str_c, 16, "%05lu", this->number);
    std::string number_str(number_str_c);
    return this->prefix + "_" + number_str + "_" + pre_suffix + this->suffix;
  }
};

inline void dump_file(const fs::path &filepath, const std::string &info)
{
  std::fstream fout;
  fout.open(filepath, std::ios::out);
  if (fout.is_open()) {
    fout << info;
    fout.close();
  }
  else {
    std::cerr << "Couldn't open file " << filepath.c_str() << std::endl;
  }
}

inline void copy_v2_float2(float *res, const float2 &v2)
{
  res[0] = v2[0];
  res[1] = v2[1];
}

inline void copy_v3_float3(float *res, const float3 &v3)
{
  res[0] = v3[0];
  res[1] = v3[1];
  res[2] = v3[2];
}

template<typename T> std::ostream &operator<<(std::ostream &stream, const blender::Vector<T> &vec)
{
  if (vec.size() == 0) {
    stream << "()";
    return stream;
  }
  stream << "(";
  for (const auto &i : vec) {
    stream << i << ", ";
  }
  stream << "\b\b)";
  return stream;
}

template<typename T> std::ostream &operator<<(std::ostream &stream, const ga::Arena<T> &arena)
{
  if (arena.size() == 0) {
    stream << "()";
    return stream;
  }
  stream << "(";
  for (const auto &i : arena) {
    stream << i << ", ";
  }
  stream << "\b\b)";
  return stream;
}

template<typename T> std::ostream &operator<<(std::ostream &stream, const std::optional<T> &option)
{
  if (option) {
    stream << "Some(" << option.value() << ")";
  }
  else {
    stream << "None";
  }
  return stream;
}

template<typename... Types>
constexpr std::ostream &operator<<(std::ostream &stream, const std::tuple<Types...> &tuple)
{
  auto tuple_size = std::tuple_size<std::tuple<Types...>>();

  if (tuple_size == 0) {
    stream << "()";
    return stream;
  }

  stream << "(";
  std::apply([&](const auto &... i) { ((stream << i << ", "), ...); }, tuple);
  stream << "\b\b)";

  return stream;
}

/**
 * `Node`: Stores the worldspace/localspace coordinates of the
 * `Mesh`. Commonly called the vertex of the mesh (note: in this mesh
 * structure `Vert` is not the commonly known vertex of the mesh, `Node` is).
 *
 * Stores information about `Vert`(s) that refer to this `Node`.
 */
template<typename T> class Node {
  NodeIndex self_index;
  IncidentVerts verts;

  float3 pos;
  float3 normal;
  std::optional<T> extra_data;

 public:
  Node(NodeIndex self_index, float3 pos, float3 normal)
      : self_index(self_index), pos(pos), normal(normal)
  {
  }

  const auto &get_self_index() const
  {
    return this->self_index;
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

  const auto &get_checked_extra_data() const
  {
    BLI_assert(this->extra_data);
    return this->extra_data.value();
  }

  auto &get_checked_extra_data_mut()
  {
    BLI_assert(this->extra_data);
    return this->extra_data.value();
  }

  const auto &get_verts() const
  {
    return this->verts;
  }

  const auto &get_normal() const
  {
    return this->normal;
  }

  friend std::ostream &operator<<(std::ostream &stream, const Node &node)
  {
    stream << "(self_index: " << node.self_index << ", verts: " << node.verts
           << ", pos: " << node.pos << ", normal: " << node.normal << ")";
    return stream;
  }

  MSGPACK_DEFINE(self_index, verts, pos, normal, extra_data);

  template<typename, typename, typename, typename> friend class Mesh;
};

/**
 * `Vert`: Stores the UV (2D) coordinates of the mesh. Acts as the glue
 * between the faces and the position of the vertices. This is needed
 * because each face can have distinct UV (2D) coordinates but same
 * position (3D) coordinations.
 *
 * Stores information about which `Edge`(s) refer to this `Vert`.
 *
 * Stores information about which `Node` to point to. This is needed
 * for the 3D coordinates of the vertices of the mesh. Refer above for
 * information about why `Vert` and `Node` are needed.
 */
template<typename T> class Vert {
  VertIndex self_index;
  IncidentEdges edges;
  std::optional<NodeIndex> node;

  float2 uv;
  std::optional<T> extra_data;

 public:
  Vert(VertIndex self_index, float2 uv) : self_index(self_index), uv(uv)
  {
  }

  const auto &get_self_index() const
  {
    return this->self_index;
  }

  const auto &get_uv() const
  {
    return this->uv;
  }

  auto &get_uv_mut()
  {
    return this->uv;
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

  const auto &get_checked_extra_data() const
  {
    BLI_assert(this->extra_data);
    return this->extra_data.value();
  }

  auto &get_checked_extra_data_mut()
  {
    BLI_assert(this->extra_data);
    return this->extra_data.value();
  }

  const auto &get_edges() const
  {
    return this->edges;
  }

  auto get_node() const
  {
    return this->node;
  }

  friend std::ostream &operator<<(std::ostream &stream, const Vert &vert)
  {
    stream << "(self_index: " << vert.self_index << ", edges: " << vert.edges
           << ", node: " << vert.node << ", uv: " << vert.uv << ")";
    return stream;
  }

  MSGPACK_DEFINE(self_index, edges, node, uv, extra_data);

  template<typename, typename, typename, typename> friend class Mesh;
};

/**
 * `Edge`: Acts as the glue between the `Face` and the `Vert`(s).
 *
 * TODO(ish): might be possible to remove this entirely to save space
 * or directly store `Node` instead of `Vert`.
 *
 * Stores information about which `Face`(s) refer to this
 * `Edge`. (note: this is a one way relation, it is possible to
 * indirectly get the `Edge` from the `Face` from the `Vert`
 * information stored in the `Face`.)
 *
 * Stores information about which `Vert`(s) (as a tuple of 2
 * `VertIndex`) refer to this `Edge`.
 */
template<typename T> class Edge {
  EdgeIndex self_index;
  IncidentFaces faces;
  std::optional<EdgeVerts> verts;

  std::optional<T> extra_data;

 public:
  Edge(EdgeIndex self_index) : self_index(self_index)
  {
  }

  const auto &get_self_index() const
  {
    return this->self_index;
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

  const auto &get_checked_extra_data() const
  {
    BLI_assert(this->extra_data);
    return this->extra_data.value();
  }

  auto &get_checked_extra_data_mut()
  {
    BLI_assert(this->extra_data);
    return this->extra_data.value();
  }

  bool has_vert(VertIndex vert_index) const
  {
    if (this->verts) {
      if (std::get<0>(this->verts.value()) == vert_index ||
          std::get<1>(this->verts.value()) == vert_index) {
        return true;
      }
    }

    return false;
  }

  bool is_loose() const
  {
    return this->faces.size() == 0;
  }

  const auto &get_faces() const
  {
    return this->faces;
  }

  const auto &get_verts() const
  {
    return this->verts;
  }

  const auto &get_checked_verts() const
  {
    BLI_assert(this->verts);
    return this->verts.value();
  }

  VertIndex get_checked_other_vert(const VertIndex &vert_index) const
  {
    BLI_assert(this->has_vert(vert_index));

    const auto &verts = this->get_checked_verts();
    if (std::get<0>(verts) == vert_index) {
      return std::get<1>(verts);
    }
    return std::get<0>(verts);
  }

  friend std::ostream &operator<<(std::ostream &stream, const Edge &edge)
  {
    stream << "(self_index: " << edge.self_index << ", faces: " << edge.faces
           << ", verts: " << edge.verts << ")";
    return stream;
  }

  MSGPACK_DEFINE(self_index, faces, verts, extra_data);

  template<typename, typename, typename, typename> friend class Mesh;
};

/**
 * `Face`: Stores the connectivity of the `Mesh`.
 *
 * Stores information about which `Vert`(s) make up this face. (note:
 * it is `Vert` instead of `Node` since there need to be seams to UV
 * unwrap the mesh and this leads to vertices of the face having
 * different UV (2D) coordinates but same position (3D) coordinates,
 * so this is split into `Vert` and `Node` respectively.)

 * Stores the face normal. This must be recomputed after any changes
 * to the Mesh. Assume dirty always.
 */
template<typename T> class Face {
  FaceIndex self_index;
  AdjacentVerts verts;

  float3 normal;
  std::optional<T> extra_data;

 public:
  Face(FaceIndex self_index, float3 normal) : self_index(self_index), normal(normal)
  {
  }

  const auto &get_self_index() const
  {
    return this->self_index;
  }

  const auto &get_normal() const
  {
    return this->normal;
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

  const auto &get_checked_extra_data() const
  {
    BLI_assert(this->extra_data);
    return this->extra_data.value();
  }

  auto &get_checked_extra_data_mut()
  {
    BLI_assert(this->extra_data);
    return this->extra_data.value();
  }

  const auto &get_verts() const
  {
    return this->verts;
  }

  bool has_vert_index(const VertIndex &vert_index) const
  {
    return verts.contains(vert_index);
  }

  template<typename EED> bool has_edge(const Edge<EED> &edge) const
  {
    BLI_assert(edge.get_verts());
    auto &[edge_vert_1, edge_vert_2] = edge.get_verts().value();

    BLI_assert(this->has_vert_index(edge_vert_1));
    BLI_assert(this->has_vert_index(edge_vert_2));

    auto vi1 = this->verts.first_index_of(edge_vert_1);
    auto vi2 = this->verts.first_index_of(edge_vert_2);

    if (std::abs(vi1 - vi2) == 1) {
      return true;
    }

    /* TODO(ish): there probably a nicer way to check for this
     * special case, this is way too verbose */
    /* Need to have loop around as well, so if the face has 5 verts,
     * verts at [0, 1, 2, 3, 4]. Then an edge (0, 4) or (4, 0) can
     * exist. Thus an extra check is necessary */
    if (vi1 == 0) {
      if (vi2 == this->verts.size() - 1) {
        return true;
      }
      return false;
    }
    if (vi2 == 0) {
      if (vi1 == this->verts.size() - 1) {
        return true;
      }
      return false;
    }

    return false;
  }

  friend std::ostream &operator<<(std::ostream &stream, const Face &face)
  {
    stream << "(self_index: " << face.self_index << ", verts: " << face.verts
           << ", normal: " << face.normal << ")";
    return stream;
  }

  MSGPACK_DEFINE(self_index, verts, normal, extra_data);

  template<typename, typename, typename, typename> friend class Mesh;
};

class MeshIO {
  using FaceData = std::tuple<usize, usize, usize>; /* position,
                                                     * uv,
                                                     * normal */

  blender::Vector<float3> positions;
  blender::Vector<float2> uvs;
  blender::Vector<float3> normals;
  blender::Vector<blender::Vector<FaceData>> face_indices;
  blender::Vector<blender::Vector<usize>> line_indices;

 public:
  enum IOType {
    IOTYPE_OBJ,
  };

  MeshIO() = default;

  bool read(const fs::path &filepath, IOType type)
  {
    if (type != IOTYPE_OBJ) {
      return false;
    }

    if (!fs::exists(filepath)) {
      return false;
    }

    std::fstream fin;
    fin.open(filepath, std::ios::in);

    if (!fin.is_open()) {
      return false;
    }

    return read(std::move(fin), type);
  }

  bool read(std::istream &&in, IOType type)
  {
    if (type == IOTYPE_OBJ) {
      auto res = this->read_obj(std::move(in));
      if (!res) {
        return false;
      }
    }
    else {
      BLI_assert_unreachable();
    }

    /* TODO(ish): do some checks to ensure the data makes sense */

    return true;
  }

  bool read(::Mesh *mesh)
  {
    BLI_assert(mesh != nullptr);

    /* Update the mesh internal pointers with the customdata stuff */
    BKE_mesh_update_customdata_pointers(mesh, false);

    if (mesh->totvert == 0) {
      return false;
    }

    /* We need uv information */
    if (mesh->mloopuv == nullptr) {
      return false;
    }

    auto &positions = this->positions;
    auto &uvs = this->uvs;
    auto &normals = this->normals;
    auto &face_indices = this->face_indices;
    auto &line_indices = this->line_indices;

    /* TODO(ish): might make sense to clear all these vectors */

    /* TODO(ish): check if normals must be recalcuated */

    for (auto i = 0; i < mesh->totvert; i++) {
      positions.append(mesh->mvert[i].co);
      float normal[3];
      normal_short_to_float_v3(normal, mesh->mvert[i].no);
      normals.append(normal);
    }

    /* A UV map is needed because the UVs stored the mesh are stored
     * per loop which means there are UVs that repeat, this leads to
     * having 2 or more unique edges per 1 true edge. The UV map is
     * used to combine UVs and create the correct indexing for these.
     *
     * The map has a key as the tuple stores the mloop::v and uv
     * coordinates. The map has the corresponding index stored.
     */
    using UVMapKey = std::pair<usize, float2>;
    blender::Map<UVMapKey, usize> uv_map;
    usize true_uv_index = 0;

    for (auto i = 0; i < mesh->totpoly; i++) {
      const auto &mp = mesh->mpoly[i];
      blender::Vector<FaceData> face;
      face.reserve(mp.totloop);

      for (auto j = 0; j < mp.totloop; j++) {
        const auto &ml = mesh->mloop[mp.loopstart + j];
        usize pos_index = ml.v;
        usize normal_index = ml.v;

        const UVMapKey key = {ml.v, mesh->mloopuv[mp.loopstart + j].uv};
        if (uv_map.contains(key) == false) {
          uvs.append(mesh->mloopuv[mp.loopstart + j].uv);
          uv_map.add_new(key, true_uv_index);
          true_uv_index++;
        }
        usize uv_index = uv_map.lookup(key);

        face.append(std::make_tuple(pos_index, uv_index, normal_index));
      }

      face_indices.append(face);
    }

    for (auto i = 0; i < mesh->totedge; i++) {
      const auto &me = mesh->medge[i];

      if (me.flag & ME_LOOSEEDGE) {
        blender::Vector<usize> line;
        line.append(me.v1);
        line.append(me.v2);
        line_indices.append(line);
      }
    }

    return true;
  }

  bool write(const fs::path &filepath, IOType type)
  {
    if (type != IOTYPE_OBJ) {
      return false;
    }

    if (!fs::exists(filepath)) {
      return false;
    }

    std::fstream fout;
    fout.open(filepath, std::ios::out);

    if (!fout.is_open()) {
      return false;
    }

    write(fout, type);

    return true;
  }

  void write(std::ostream &out, IOType type)
  {
    if (type == IOTYPE_OBJ) {
      this->write_obj(out);
    }
    else {
      BLI_assert_unreachable();
    }
  }

  ::Mesh *write()
  {
    auto num_verts = this->positions.size();
    auto num_edges = 0;
    for (const auto &line : this->line_indices) {
      num_edges += line.size() - 1;
    }
    auto num_loops = 0;
    for (const auto &face : this->face_indices) {
      num_loops += face.size();
    }
    auto num_uvs = num_loops; /* for `::Mesh` the number of uvs has
                               * to match number of loops  */
    auto num_poly = this->face_indices.size();
    auto *mesh = BKE_mesh_new_nomain(num_verts, num_edges, 0, num_loops, num_poly);
    if (!mesh) {
      return nullptr;
    }

    CustomData_add_layer(&mesh->ldata, CD_MLOOPUV, CD_CALLOC, nullptr, num_uvs);

    BKE_mesh_update_customdata_pointers(mesh, false);

    auto *mverts = mesh->mvert;
    auto *medges = mesh->medge;
    auto *mloopuvs = mesh->mloopuv;
    auto *mloops = mesh->mloop;
    auto *mpolys = mesh->mpoly;

    for (auto i = 0; i < this->positions.size(); i++) {
      copy_v3_float3(mverts[i].co, this->positions[i]);
    }

    auto edges_total = 0;
    for (auto i = 0; i < this->line_indices.size(); i++) {
      const auto &edge = this->line_indices[i];
      for (auto j = 0; j < edge.size() - 1; j++) {
        auto &me = medges[edges_total + j];
        me.v1 = edge[j];
        me.v2 = edge[j + 1];
        me.flag |= ME_LOOSEEDGE;
      }

      edges_total += edge.size() - 1;
    }

    auto loopstart = 0;
    for (auto i = 0; i < this->face_indices.size(); i++) {
      const auto &face = this->face_indices[i];
      auto &mpoly = mpolys[i];
      mpoly.loopstart = loopstart;
      mpoly.totloop = face.size();

      for (auto j = 0; j < face.size(); j++) {
        auto [pos_index, uv_index, normal_index] = face[j];
        /* TODO(ish): handle normal index */
        mloops[loopstart + j].v = pos_index;

        /* Need to update mloopuvs here since `mesh->mloop` and
         * `mesh->mloopuv` need to maintain same size and correspond
         * with one another  */
        copy_v2_float2(mloopuvs[loopstart + j].uv, this->uvs[uv_index]);
      }

      loopstart += face.size();
    }

    BKE_mesh_ensure_normals(mesh);
    BKE_mesh_calc_edges(mesh, true, false);

    /* TODO(ish): handle vertex normals */

    return mesh;
  }

  void set_positions(blender::Vector<float3> &&positions)
  {
    this->positions = std::move(positions);
  }

  void set_uvs(blender::Vector<float2> &&uvs)
  {
    this->uvs = std::move(uvs);
  }

  void set_normals(blender::Vector<float3> &&normals)
  {
    this->normals = std::move(normals);
  }

  void set_face_indices(blender::Vector<blender::Vector<FaceData>> &&face_indices)
  {
    this->face_indices = std::move(face_indices);
  }

  void set_line_indices(blender::Vector<blender::Vector<usize>> &&line_indices)
  {
    this->line_indices = std::move(line_indices);
  }

  const auto &get_positions() const
  {
    return this->positions;
  }

  const auto &get_uvs() const
  {
    return this->uvs;
  }

  const auto &get_normals() const
  {
    return this->normals;
  }

  const auto &get_face_indices() const
  {
    return this->face_indices;
  }

  const auto &get_line_indices() const
  {
    return this->line_indices;
  }

  static constexpr inline auto invalid_index()
  {
    return std::numeric_limits<usize>::max();
  }

  friend std::ostream &operator<<(std::ostream &stream, const MeshIO &meshio)
  {
    stream << "positions: " << meshio.get_positions() << std::endl;
    stream << "uvs: " << meshio.get_uvs() << std::endl;
    stream << "normals: " << meshio.get_normals() << std::endl;
    stream << "face_indices: " << meshio.get_face_indices() << std::endl;
    stream << "line_indices: " << meshio.get_line_indices();
    return stream;
  }

 private:
  blender::Vector<std::string> tokenize(std::string const &str, const char delim)
  {
    blender::Vector<std::string> res;
    // construct a stream from the string
    std::stringstream ss(str);

    std::string s;
    while (std::getline(ss, s, delim)) {
      res.append(s);
    }

    return res;
  }

  bool read_obj(std::istream &&in)
  {
    std::string line;
    while (std::getline(in, line)) {
      if (line.rfind('#', 0) == 0) {
        continue;
      }

      if (line.rfind("v ", 0) == 0) {
        std::istringstream li(line);
        float x, y, z;
        std::string temp;
        li >> temp >> x >> y >> z;
        if (li.fail()) {
          return false;
        }
        BLI_assert(temp == "v");
        this->positions.append(float3(x, y, z));
      }
      else if (line.rfind("vt ", 0) == 0) {
        std::istringstream li(line);
        float u, v;
        std::string temp;
        li >> temp >> u >> v;
        if (li.fail()) {
          return false;
        }
        BLI_assert(temp == "vt");
        this->uvs.append(float2(u, v));
      }
      else if (line.rfind("vn ", 0) == 0) {
        std::istringstream li(line);
        float x, y, z;
        std::string temp;
        li >> temp >> x >> y >> z;
        if (li.fail()) {
          return false;
        }
        BLI_assert(temp == "vn");
        this->normals.append(float3(x, y, z));
      }
      else if (line.rfind("f ", 0) == 0) {
        const auto line_toks = this->tokenize(line, ' ');

        BLI_assert(line_toks.size() != 0);

        blender::Vector<FaceData> face;

        for (const auto *indices_group_iter = line_toks.begin() + 1;
             indices_group_iter != line_toks.end();
             indices_group_iter++) {
          const auto indices_group = *indices_group_iter;

          auto indices_str = this->tokenize(indices_group, '/');
          if (indices_str.size() == 1) {
            std::istringstream isi(indices_str[0]);
            usize pos_index;
            isi >> pos_index;
            face.append(
                std::make_tuple(pos_index - 1, MeshIO::invalid_index(), MeshIO::invalid_index()));
          }
          else if (indices_str.size() == 2) {
            std::istringstream isi_pos(indices_str[0]);
            std::istringstream isi_uv(indices_str[1]);
            usize pos_index;
            usize uv_index;
            isi_pos >> pos_index;
            isi_uv >> uv_index;
            face.append(std::make_tuple(pos_index - 1, uv_index - 1, MeshIO::invalid_index()));
          }
          else if (indices_str.size() == 3) {
            std::istringstream isi_pos(indices_str[0]);
            std::istringstream isi_uv(indices_str[1]);
            std::istringstream isi_normal(indices_str[2]);
            usize pos_index;
            usize uv_index;
            usize normal_index;
            isi_pos >> pos_index;
            isi_uv >> uv_index;
            isi_normal >> normal_index;
            face.append(std::make_tuple(pos_index - 1, uv_index - 1, normal_index - 1));
          }
          else {
            return false;
          }
        }

        BLI_assert(line_toks[0] == "f");
        this->face_indices.append(face);
      }
      else if (line.rfind("l ", 0) == 0) {
        std::istringstream li(line);
        std::string temp;
        li >> temp;

        blender::Vector<usize> indices;
        usize index;
        while (li >> index) {
          indices.append(index - 1); /* obj starts from 1, we want to
                                      * start from 0 */
        }

        BLI_assert(temp == "l");
        this->line_indices.append(indices);
      }
      else {
        /* unknown type, continuing */
        continue;
      }
    }

    return true;
  }

  void write_obj(std::ostream &out)
  {
    for (const auto &pos : this->positions) {
      out << "v " << pos.x << " " << pos.y << " " << pos.z << std::endl;
    }

    for (const auto &uv : this->uvs) {
      out << "vt " << uv.x << " " << uv.y << std::endl;
    }

    for (const auto &normal : this->normals) {
      out << "vn " << normal.x << " " << normal.y << " " << normal.z << std::endl;
    }

    for (const auto &face : this->face_indices) {
      out << "f ";
      for (const auto &[pos_index, uv_index, normal_index] : face) {
        if (normal_index == MeshIO::invalid_index()) {
          if (uv_index == MeshIO::invalid_index()) {
            out << pos_index + 1 << " ";
          }
          else {
            out << pos_index + 1 << "/" << uv_index + 1 << " ";
          }
        }
        else {
          out << pos_index + 1 << "/" << uv_index + 1 << "/" << normal_index + 1 << " ";
        }
      }
      out << std::endl;
    }

    for (const auto &line : this->line_indices) {
      out << "l ";
      for (const auto &index : line) {
        out << index + 1 << " ";
      }
      out << std::endl;
    }
  }
};

template<typename END, typename EVD, typename EED, typename EFD> class Mesh {
  /* using declarations */
  /* static data members */
  /* non-static data members */
  ga::Arena<Node<END>> nodes;
  ga::Arena<Vert<EVD>> verts;
  ga::Arena<Edge<EED>> edges;
  ga::Arena<Face<EFD>> faces;

  bool node_normals_dirty;
  bool face_normals_dirty;

 public:
  /* default constructor */
  Mesh() = default;

  /* other constructors */
  /* copy constructor */
  /* move constructor */

  /* destructor */

  /* copy assignment operator */
  /* move assignment operator */
  /* other operator overloads */
  friend std::ostream &operator<<(std::ostream &stream, const Mesh &mesh)
  {
    stream << "nodes: " << mesh.nodes << std::endl;
    stream << "verts: " << mesh.verts << std::endl;
    stream << "edges: " << mesh.edges << std::endl;
    stream << "faces: " << mesh.faces << std::endl;
    return stream;
  }

  /* all public static methods */
  /* all public non-static methods */
  const auto &get_nodes() const
  {
    return this->nodes;
  }

  auto &get_nodes_mut()
  {
    return this->nodes;
  }

  const auto &get_verts() const
  {
    return this->verts;
  }

  auto &get_verts_mut()
  {
    return this->verts;
  }

  const auto &get_edges() const
  {
    return this->edges;
  }

  auto &get_edges_mut()
  {
    return this->edges;
  }

  const auto &get_faces() const
  {
    return this->faces;
  }

  auto &get_faces_mut()
  {
    return this->faces;
  }

  bool has_node(const NodeIndex node_index) const
  {
    const auto op_node = this->get_nodes().get(node_index);
    return op_node != std::nullopt;
  }

  bool has_vert(const VertIndex vert_index) const
  {
    const auto op_vert = this->get_verts().get(vert_index);
    return op_vert != std::nullopt;
  }

  bool has_edge(const EdgeIndex edge_index) const
  {
    const auto op_edge = this->get_edges().get(edge_index);
    return op_edge != std::nullopt;
  }

  bool has_face(const FaceIndex face_index) const
  {
    const auto op_face = this->get_faces().get(face_index);
    return op_face != std::nullopt;
  }

  std::optional<EdgeIndex> get_connecting_edge_index(VertIndex vert_1_index,
                                                     VertIndex vert_2_index) const
  {
    auto op_vert_1 = this->verts.get(vert_1_index);
    if (op_vert_1 == std::nullopt) {
      return std::nullopt;
    }

    auto vert_1 = op_vert_1.value().get();

    for (const auto &edge_index : vert_1.edges) {
      auto op_edge = this->edges.get(edge_index);

      if (op_edge == std::nullopt) {
        return std::nullopt;
      }

      auto edge = op_edge.value().get();

      if (edge.has_vert(vert_2_index)) {
        return edge_index;
      }
    }

    return std::nullopt;
  }

  blender::Vector<EdgeIndex> get_connecting_edge_indices(const Node<END> &node_1,
                                                         const Node<END> &node_2) const
  {
    blender::Vector<EdgeIndex> res;
    for (const auto &vert_index_1 : node_1.get_verts()) {
      for (const auto &vert_index_2 : node_2.get_verts()) {
        auto op_edge_index = this->get_connecting_edge_index(vert_index_1, vert_index_2);
        if (op_edge_index) {
          res.append(op_edge_index.value());
        }
      }
    }

    return res;
  }

  /**
   * A "3D edge" is the set of all edges between 2 nodes. An edge
   * stores the edge verts, these are used to find the edge nodes and
   * all connecting edge indicies between the edge nodes constitute
   * the 3D edge.
   */
  blender::Vector<EdgeIndex> get_checked_3d_edge(const Edge<EED> &edge) const
  {
    const auto [n1, n2] = this->get_checked_nodes_of_edge(edge);
    return this->get_connecting_edge_indices(n1, n2);
  }

  /**
   * Gives first vert index of face that is not part of edge.
   * This should be called only when the face has 3 verts, will return
   * `std::nullopt` otherwise.
   **/
  inline std::optional<VertIndex> get_other_vert_index(EdgeIndex edge_index,
                                                       FaceIndex face_index) const
  {
    const auto &face = this->get_checked_face(face_index);

    if (face.verts.size() != 3) {
      return std::nullopt;
    }

    const auto vert_1_index = face.verts[0];
    const auto vert_2_index = face.verts[1];
    const auto vert_3_index = face.verts[2];

    const auto &edge = this->get_checked_edge(edge_index);

    /* The edge must contain the face */
    if (edge.faces.contains(face_index) == false) {
      return std::nullopt;
    }

    if (edge.has_vert(vert_1_index) == false) {
      return vert_1_index;
    }
    if (edge.has_vert(vert_2_index) == false) {
      return vert_2_index;
    }

    /* Since the edge constains the face, and since v1 and v2 are
     * part of the edge, v3 has to not be in the edge */
    BLI_assert(edge.has_vert(vert_3_index) == false);
    return vert_3_index;
  }

  /**
   * Same as unchecked version but instead of returning `std::nullopt`
   * when it fails, the checks are asserted so in debug mode, it will
   * assert and abort but in release mode it will have undefined
   * behavior.
   */
  inline VertIndex get_checked_other_vert_index(EdgeIndex edge_index, FaceIndex face_index) const
  {
    const auto &face = this->get_checked_face(face_index);
    BLI_assert(face.verts.size() == 3);

    const auto vert_1_index = face.verts[0];
    const auto vert_2_index = face.verts[1];
    const auto vert_3_index = face.verts[2];

    const auto &edge = this->get_checked_edge(edge_index);

    BLI_assert(edge.faces.contains(face_index));

    if (edge.has_vert(vert_1_index) == false) {
      return vert_1_index;
    }
    if (edge.has_vert(vert_2_index) == false) {
      return vert_2_index;
    }

    BLI_assert(edge.has_vert(vert_3_index) == false);
    return vert_3_index;
  }

  void read(const MeshIO &reader)
  {
    const auto &positions = reader.get_positions();
    const auto &uvs = reader.get_uvs();
    const auto &normals = reader.get_normals();
    const auto &face_indices = reader.get_face_indices();
    const auto &line_indices = reader.get_line_indices();

    /* TODO(ish): add support for when uvs doesn't exist */
    BLI_assert(uvs.size() != 0);

    this->node_normals_dirty = true;
    this->face_normals_dirty = true;

    /* create all `Node`(s) */
    for (const auto &pos : positions) {
      this->add_empty_node(pos, float3_unknown());
    }

    /* create all `Vert`(s) */
    for (const auto &uv : uvs) {
      this->add_empty_vert(uv);
    }

    /* use face information for create `Face`(s), `Edge`(s) and
     * create the necessary references */
    for (const auto &face_index_data : face_indices) {

      /* update verts and nodes */
      for (const auto &[pos_index, uv_index, normal_index] : face_index_data) {
        auto op_vert = this->verts.get_no_gen(uv_index);
        auto op_node = this->nodes.get_no_gen(pos_index);
        BLI_assert(op_vert && op_node);

        auto &vert = op_vert.value().get();
        auto &node = op_node.value().get();

        vert.node = node.self_index;
        node.verts.append_non_duplicates(vert.self_index);

        /* if vertex normals exist */
        if (normals.size() > normal_index) {
          node.normal = normals[normal_index];
        }
      }

      /* update edges */
      auto vert_1_i = std::get<1>(face_index_data[0]);
      auto vert_2_i = std::get<1>(face_index_data[0]);
      blender::Vector<VertIndex> face_verts;
      blender::Vector<EdgeIndex> face_edges;
      for (auto i = 1; i <= face_index_data.size(); i++) {
        vert_1_i = vert_2_i;
        if (i == face_index_data.size()) {
          vert_2_i = std::get<1>(face_index_data[0]);
        }
        else {
          vert_2_i = std::get<1>(face_index_data[i]);
        }

        auto op_vert_1_index = this->verts.get_no_gen_index(vert_1_i);
        auto op_vert_2_index = this->verts.get_no_gen_index(vert_2_i);
        BLI_assert(op_vert_1_index && op_vert_2_index);

        auto vert_1_index = op_vert_1_index.value();
        auto vert_2_index = op_vert_2_index.value();

        if (auto op_edge_index = this->get_connecting_edge_index(vert_1_index, vert_2_index)) {
          face_edges.append(op_edge_index.value());
        }
        else {
          auto &edge = this->add_empty_edge();

          edge.verts = std::make_tuple(vert_1_index, vert_2_index);

          auto &vert_1 = this->verts.get(vert_1_index).value().get();
          vert_1.edges.append(edge.self_index);

          auto &vert_2 = this->verts.get(vert_2_index).value().get();
          vert_2.edges.append(edge.self_index);

          face_edges.append(edge.self_index);
        }

        face_verts.append(vert_1_index);
      }

      /* update faces */
      {
        auto &face = this->add_empty_face(float3_unknown());

        face.verts = std::move(face_verts);

        for (const auto &edge_index : face_edges) {
          auto op_edge = this->edges.get(edge_index);
          BLI_assert(op_edge);

          auto &edge = op_edge.value().get();

          edge.faces.append(face.self_index);
        }
      }
    }

    /* Add loose edges
     *
     * Need to create "empty" `Vert`(s) since `Edge`(s) store
     * `Vert`(s) which has UV coords but since these are loose edges,
     * they don't have any UV coords. Ideally we would store `Node`(s)
     * directly but this might lead to problems else where with
     * respect to speed (the link between edges and faces would be
     * slower to access).
     **/
    {
      /* For each loose edge, add the correct connectivity */
      for (const auto &line : line_indices) {
        blender::Vector<std::tuple<usize, usize>> line_pairs;
        for (auto i = 0; i < line.size() - 1; i++) {
          line_pairs.append(std::make_tuple(line[i], line[i + 1]));
        }

        for (const auto &[node_1_i, node_2_i] : line_pairs) {
          auto &edge = this->add_empty_edge();

          auto op_node_1 = this->nodes.get_no_gen(node_1_i);
          BLI_assert(op_node_1);
          auto &node_1 = op_node_1.value().get();

          auto op_node_2 = this->nodes.get_no_gen(node_2_i);
          BLI_assert(op_node_2);
          auto &node_2 = op_node_2.value().get();

          /* For `Node`(s) without `Vert`(s) create an empty vert */
          {
            auto connect_node_to_empty_vert = [this](auto &node) {
              if (node.verts.size() == 0) {
                auto &vert = this->add_empty_vert(float2_unknown());
                vert.node = node.self_index;
                node.verts.append(vert.self_index);
              }
            };

            connect_node_to_empty_vert(node_1);
            connect_node_to_empty_vert(node_2);
          }

          /* TODO(ish): in case `node_1` or `node_2` are part of a
           * face, then it might make sense to pick the "best"
           * `Vert`, maybe via distance?  */
          /* Pick the first `Vert` of the `Node` to connect to loose edge */
          auto op_vert_1 = this->verts.get(node_1.verts[0]);
          BLI_assert(op_vert_1);
          auto &vert_1 = op_vert_1.value().get();
          auto op_vert_2 = this->verts.get(node_2.verts[0]);
          BLI_assert(op_vert_2);
          auto &vert_2 = op_vert_2.value().get();

          /* Connect up the edge with the verts */
          vert_1.edges.append(edge.self_index);
          vert_2.edges.append(edge.self_index);
          edge.verts = std::make_tuple(vert_1.self_index, vert_2.self_index);
        }
      }
    }

    /* TODO(ish): ensure normal information properly, right now need
     * to just assume it is not dirty for faster development */
    this->node_normals_dirty = false;
  }

  MeshIO write() const
  {
    using FaceData = std::tuple<usize, usize, usize>;
    blender::Vector<float3> positions;
    blender::Vector<float2> uvs;
    blender::Vector<float3> normals;
    blender::Vector<blender::Vector<FaceData>> face_indices;
    blender::Vector<blender::Vector<usize>> line_indices;

    /* To handle gaps in the arena which can lead to wrong index
     * values, a `blender::Map` is created between the
     * `node.self_index` and the corresponding true index in the
     * `positions` vector
     *
     * Same for the verts/uvs as well */

    /* TODO(ish): this assert should change to some sort of error
     * handled thing */
    BLI_assert(this->node_normals_dirty == false);

    blender::Map<NodeIndex, usize> pos_index_map;
    blender::Map<VertIndex, usize> uv_index_map;

    auto true_pos_index = 0;
    for (const auto &node : this->nodes) {
      auto pos = node.pos;
      /* dont need unkown check for position, it should always be present */
      positions.append(pos);

      auto normal = node.normal;
      if (float3_is_unknown(normal) == false) {
        normals.append(normal);
      }

      /* add the index info to the map */
      pos_index_map.add_new(node.self_index, true_pos_index);
      true_pos_index++;
    }
    BLI_assert(true_pos_index == this->nodes.size());

    auto true_uv_index = 0;
    for (const auto &vert : this->verts) {
      auto uv = vert.uv;
      if (float2_is_unknown(uv) == false) {
        uvs.append(uv);
      }

      /* add the index info to the map */
      uv_index_map.add_new(vert.self_index, true_uv_index);
      true_uv_index++;
    }
    BLI_assert(true_uv_index == this->verts.size());

    for (const auto &face : this->faces) {
      blender::Vector<FaceData> io_face;

      for (const auto &vert_index : face.verts) {
        auto op_vert = this->verts.get(vert_index);
        BLI_assert(op_vert);
        const auto &vert = op_vert.value().get();

        BLI_assert(vert.node); /* a vert cannot exist without a node */

        /* get the indices from the index maps */
        auto pos_index = pos_index_map.lookup(vert.node.value());
        auto uv_index = uv_index_map.lookup(vert.self_index);
        auto normal_index = pos_index;

        io_face.append(std::make_tuple(pos_index, uv_index, normal_index));
      }

      face_indices.append(io_face);
    }

    for (const auto &edge : this->edges) {
      if (edge.is_loose()) {
        blender::Vector<usize> line;

        BLI_assert(edge.verts);

        const auto &vert_indices = edge.verts.value();

        const auto op_vert_1 = this->verts.get(std::get<0>(vert_indices));
        const auto op_vert_2 = this->verts.get(std::get<1>(vert_indices));

        BLI_assert(op_vert_1);
        BLI_assert(op_vert_2);

        const auto &vert_1 = op_vert_1.value().get();
        const auto &vert_2 = op_vert_2.value().get();

        const auto op_node_1_index = vert_1.node;
        const auto op_node_2_index = vert_2.node;

        BLI_assert(op_node_1_index);
        BLI_assert(op_node_2_index);

        const auto node_1_index = op_node_1_index.value();
        const auto node_2_index = op_node_2_index.value();

        auto pos_index_1 = pos_index_map.lookup(node_1_index);
        auto pos_index_2 = pos_index_map.lookup(node_2_index);

        line.append(pos_index_1);
        line.append(pos_index_2);

        line_indices.append(line);
      }
    }

    MeshIO result;
    result.set_positions(std::move(positions));
    result.set_uvs(std::move(uvs));
    result.set_normals(std::move(normals));
    result.set_face_indices(std::move(face_indices));
    result.set_line_indices(std::move(line_indices));

    return result;
  }

  void read_obj(const fs::path &filepath)
  {
    MeshIO reader;
    const auto reader_success = reader.read(filepath, MeshIO::IOTYPE_OBJ);
    BLI_assert(reader_success); /* must successfully load obj */

    this->read(reader);
  }

  /**
   * Splits the edge and keeps the triangulation of the Mesh
   *
   * @param across_seams If true, think of edge as world space edge
   * and not UV space, this means, all the faces across all the edges
   * formed between the nodes of the given edge are also split and
   * triangulated regardless if it on a seam or not.
   *
   * @param copy_extra_data_for_split_edge If true, the extra data
   * from the edge that is split to form 2 new edges is copied into
   * the 2 new edges. The extra data is not copied to the edges that
   * are added to ensure triangulation.
   *
   * Returns the `MeshDiff` that lead to the operation.
   *
   * Note, the caller must ensure the adjacent faces to the edge are
   * triangulated. In debug mode, it will assert, in release mode, it
   * is undefined behaviour.
   **/
  MeshDiff<END, EVD, EED, EFD> split_edge_triangulate(EdgeIndex edge_index,
                                                      bool across_seams,
                                                      bool copy_extra_data_for_split_edge)
  {
    /* This operation will delete the following-
     * the edge specified, faces incident to the edge.
     *
     * This operation will add the following-
     * a new vert interpolated using the edge's verts, new node
     * interpolated using the node of the edge's verts.
     */

    blender::Vector<NodeIndex> added_nodes;
    blender::Vector<VertIndex> added_verts;
    blender::Vector<EdgeIndex> added_edges;
    blender::Vector<FaceIndex> added_faces;
    blender::Vector<Node<END>> deleted_nodes;
    blender::Vector<Vert<EVD>> deleted_verts;
    blender::Vector<Edge<EED>> deleted_edges;
    blender::Vector<Face<EFD>> deleted_faces;

    auto &edge_pre = this->get_checked_edge(edge_index);
    auto [edge_vert_1_pre, edge_vert_2_pre] = this->get_checked_verts_of_edge(edge_pre, false);
    auto &edge_node_1 = this->get_checked_node_of_vert(edge_vert_1_pre);
    auto &edge_node_2 = this->get_checked_node_of_vert(edge_vert_2_pre);

    blender::Vector<EdgeIndex> edge_indices = {edge_index};
    if (across_seams) {
      edge_indices = this->get_connecting_edge_indices(edge_node_1, edge_node_2);
    }

    /* Create the new new by interpolating the nodes of the edge */
    auto &new_node = this->add_empty_interp_node(edge_node_1, edge_node_2);
    added_nodes.append(new_node.self_index);

    for (const auto &edge_index : edge_indices) {
      auto &edge_a = this->get_checked_edge(edge_index);
      auto [edge_vert_1_a, edge_vert_2_a] = this->get_checked_verts_of_edge(edge_a, false);
      const auto orig_edge_extra_data = edge_a.get_extra_data();

      /* Create the new vert by interpolating the verts of the edge */
      auto &new_vert = this->add_empty_interp_vert(edge_vert_1_a, edge_vert_2_a);
      added_verts.append(new_vert.self_index);

      auto [edge_vert_1_b, edge_vert_2_b] = this->get_checked_verts_of_edge(edge_a, false);

      /* Link new_vert with new_node */
      new_vert.node = new_node.self_index;
      new_node.verts.append(new_vert.self_index);

      /* Create edges between edge_vert_1, new_vert, edge_vert_2 */
      auto &new_edge_1 = this->add_empty_edge();
      new_edge_1.verts = {edge_vert_1_b.self_index, new_vert.self_index};
      added_edges.append(new_edge_1.self_index);
      auto new_edge_1_index = new_edge_1.self_index;
      this->add_edge_ref_to_verts(new_edge_1);
      if (copy_extra_data_for_split_edge) {
        auto &extra_data = new_edge_1.get_extra_data_mut();
        extra_data = orig_edge_extra_data;
      }

      auto &new_edge_2 = this->add_empty_edge();
      new_edge_2.verts = {new_vert.self_index, edge_vert_2_b.self_index};
      added_edges.append(new_edge_2.self_index);
      auto new_edge_2_index = new_edge_2.self_index;
      this->add_edge_ref_to_verts(new_edge_2);
      if (copy_extra_data_for_split_edge) {
        auto &extra_data = new_edge_2.get_extra_data_mut();
        extra_data = orig_edge_extra_data;
      }

      /* Need to reinitialize edge because `add_empty_edge()` may have
       * reallocated `this->edges` */
      auto &edge_b = this->get_checked_edge(edge_index);
      auto faces = edge_b.faces;

      for (const auto &face_index : faces) {
        this->delink_face_edges(face_index);
        auto face = this->delete_face(face_index);

        /* Ensure the faces are triangulated before calling this function */
        BLI_assert(face.verts.size() == 3);

        auto &edge_c = this->get_checked_edge(edge_index);
        auto &other_vert = this->get_checked_other_vert(edge_c, face);

        /* Handle new face and new edge creation */
        {
          /* Handle new edge creation between new_vert and other_vert */
          auto &new_edge = this->add_empty_edge();
          new_edge.verts = std::make_tuple(other_vert.self_index, new_vert.self_index);
          added_edges.append(new_edge.self_index);
          this->add_edge_ref_to_verts(new_edge);

          auto &new_face_1 = this->add_empty_face(face.normal);
          /* Set correct orientation by swapping ev2 for nv */
          new_face_1.verts = face.verts;
          new_face_1.verts[new_face_1.verts.first_index_of(edge_vert_2_b.self_index)] =
              new_vert.self_index;
          added_faces.append(new_face_1.self_index);

          /* link edges with new_face_1 */
          {
            new_edge.faces.append(new_face_1.self_index);
            auto &new_edge_1 = this->get_checked_edge(new_edge_1_index);
            new_edge_1.faces.append(new_face_1.self_index);
            auto old_edge_1_index = this->get_connecting_edge_index(other_vert.self_index,
                                                                    edge_vert_1_b.self_index)
                                        .value();
            auto &old_edge_1 = this->get_checked_edge(old_edge_1_index);
            old_edge_1.faces.append(new_face_1.self_index);
          }

          /* Here `face` doesn't need to be reinitialized because this
           * for loop owns `face` */

          auto &new_face_2 = this->add_empty_face(face.normal);
          /* Set correct orientation by swapping ev1 for nv */
          new_face_2.verts = face.verts;
          new_face_2.verts[new_face_2.verts.first_index_of(edge_vert_1_b.self_index)] =
              new_vert.self_index;
          added_faces.append(new_face_2.self_index);

          /* link edges with new_face_2 */
          {
            new_edge.faces.append(new_face_2.self_index);
            auto &new_edge_2 = this->get_checked_edge(new_edge_2_index);
            new_edge_2.faces.append(new_face_2.self_index);
            auto old_edge_2_index = this->get_connecting_edge_index(other_vert.self_index,
                                                                    edge_vert_2_b.self_index)
                                        .value();
            auto &old_edge_2 = this->get_checked_edge(old_edge_2_index);
            old_edge_2.faces.append(new_face_2.self_index);
          }

          /* Here `face` doesn't need to be reinitialized because this
           * for loop owns `face` */
          deleted_faces.append(std::move(face));
        }
      }

      auto edge_c = this->delete_edge(edge_index);

      deleted_edges.append(std::move(edge_c));
    }
    return MeshDiff(std::move(added_nodes),
                    std::move(added_verts),
                    std::move(added_edges),
                    std::move(added_faces),
                    std::move(deleted_nodes),
                    std::move(deleted_verts),
                    std::move(deleted_edges),
                    std::move(deleted_faces));
  }

  bool is_edge_collapseable(EdgeIndex edge_index, bool verts_swapped, bool across_seams) const
  {
    /* The edge is always collapseable if across seams is false */
    if (across_seams == false) {
      return true;
    }

    const auto &e_a = this->get_checked_edge(edge_index);
    const auto [v1_a, v2_a] = this->get_checked_verts_of_edge(e_a, verts_swapped);
    const auto &n1_a = this->get_checked_node_of_vert(v1_a);
    const auto &n2_a = this->get_checked_node_of_vert(v2_a);
    const auto n1_index = n1_a.self_index;
    const auto edge_indices = this->get_connecting_edge_indices(n1_a, n2_a);

    /* The collapse edge function doesn't support collapsing one v1 into
     * multiple v2 as of right now, so if we find such a case tell
     * user that edge is not collapseable */
    {
      blender::Vector<VertIndex> v1_list;
      for (const auto &edge_index : edge_indices) {
        const auto &e = this->get_checked_edge(edge_index);
        const auto [v1_index, v2_index] = this->get_checked_vert_indices_of_edge_aligned_with_n1(
            e, n1_index);

        if (v1_list.contains(v1_index)) {
          return false;
        }

        v1_list.append(v1_index);
      }
    }

    return true;
  }

  /**
   * Collapses the edge from edge v1 to v2 unless `verts_swapped` is set
   * to true and keeps the triangulation of the Mesh
   *
   * @param verts_swapped If true, then the edge collapsed from v2 to
   * v1, if false, edge is collapsed from v1 to v2.
   *
   * @param across_seams If true, think of edge as world space edge
   * and not UV space, this means, all the faces across all the edges
   * formed between the nodes of the given edge are also split and
   * triangulated regardless if it on a seam or not.
   *
   * Returns the `MeshDiff` that lead to the operation.
   *
   * Note, the caller must ensure the adjacent faces to the edge are
   * triangulated. In debug mode, it will assert, in release mode, it
   * is undefined behaviour.
   *
   * Caller must ensure that the edge is collapseable by calling `is_edge_collapseable()`
   **/
  MeshDiff<END, EVD, EED, EFD> collapse_edge_triangulate(EdgeIndex edge_index,
                                                         bool verts_swapped,
                                                         bool across_seams)
  {
    /* TODO(ish): write the below thing */
    /* Let the vert remove be `v1`, node to remove be `n1`, the other
     * vert, node be `v2`, `n2`.
     *
     * Let `e` be the edge to collapse.
     *
     * Let `f` be any face of `e`.
     *
     * Let `ov` be the other vert of `e`, `f`.
     *
     * Let `vx` be other vert of the edges of `v1`.
     */
    /* This operation will delete the following-
     * `v1`
     * `n1` if needed
     * `f`
     * edge between `ov` and `v1` if needed
     *
     * This operation will add the following-
     * None
     */

    BLI_assert(this->is_edge_collapseable(edge_index, verts_swapped, across_seams));

    blender::Vector<NodeIndex> added_nodes;
    blender::Vector<VertIndex> added_verts;
    blender::Vector<EdgeIndex> added_edges;
    blender::Vector<FaceIndex> added_faces;
    blender::Vector<Node<END>> deleted_nodes;
    blender::Vector<Vert<EVD>> deleted_verts;
    blender::Vector<Edge<EED>> deleted_edges;
    blender::Vector<Face<EFD>> deleted_faces;

    auto &e_a = this->get_checked_edge(edge_index);
    auto [v1_a, v2_a] = this->get_checked_verts_of_edge(e_a, verts_swapped);
    auto &n1_a = this->get_checked_node_of_vert(v1_a);
    auto &n2_a = this->get_checked_node_of_vert(v2_a);
    auto n1_index = n1_a.self_index;
    auto n2_index = n2_a.self_index;

    blender::Vector<EdgeIndex> edge_indices = {edge_index};
    if (across_seams) {
      edge_indices = this->get_connecting_edge_indices(n1_a, n2_a);
    }

    for (const auto &edge_index : edge_indices) {
      auto &e = this->get_checked_edge(edge_index);
      auto [v1_index, v2_index] = this->get_checked_vert_indices_of_edge_aligned_with_n1(e,
                                                                                         n1_index);

      auto v1_face_indices = this->get_checked_face_indices_of_vert(v1_index);

      /* Create the new faces by swapping v1 with v2 */
      {
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

          // Create the edges between v2 and the other verts
          {
            for (const auto &vert_index : vert_indices) {
              if (vert_index == v2_index) {
                continue;
              }
              /* It is possible to have a connecting edge between
               * vert_index and v2_index, in case of this, don't create
               * a new edge */
              if (this->get_connecting_edge_index(vert_index, v2_index)) {
                continue;
              }
              auto &new_e = this->add_empty_edge();
              new_e.verts = {v2_index, vert_index};
              this->add_edge_ref_to_verts(new_e);

              added_edges.append(new_e.self_index);
            }
          }

          auto &new_f = this->add_face_triangulated(
              vert_indices[0], vert_indices[1], vert_indices[2], f.normal);

          added_faces.append(new_f.self_index);
        }
      }
    }

    /* There can be multiple v2, so cannot delete the all edges or
     * faces around v1 in the previous loop */
    {
      blender::Set<VertIndex> to_delete_vert_indices;
      blender::Set<VertIndex> to_delete_edge_indices;
      blender::Set<VertIndex> to_delete_face_indices;

      for (const auto &edge_index : edge_indices) {
        const auto &e = this->get_checked_edge(edge_index);
        const auto [v1_index, v2_index] = this->get_checked_vert_indices_of_edge_aligned_with_n1(
            e, n1_index);

        auto v1_face_indices = this->get_checked_face_indices_of_vert(v1_index);

        for (const auto &face_index : v1_face_indices) {
          to_delete_face_indices.add(face_index);
        }

        to_delete_vert_indices.add(v1_index);

        to_delete_edge_indices.add(edge_index);

        const auto &v1 = this->get_checked_vert(v1_index);
        for (const auto &e_index : v1.get_edges()) {
          to_delete_edge_indices.add(e_index);
        }

        /* If any edges around v2 that have only one face and that
         * face contains v1, it will become a loose edge, so delete
         * it */
        const auto &v2 = this->get_checked_vert(v2_index);
        for (const auto &v2_e_index : v2.get_edges()) {
          const auto &v2_e = this->get_checked_edge(v2_e_index);
          if (v2_e.get_faces().size() == 1) {
            const auto &v2_e_f = this->get_checked_face(v2_e.get_faces()[0]);

            if (v2_e_f.has_vert_index(v1_index)) {
              to_delete_edge_indices.add(v2_e_index);

              /* It is possible for `ov` to become a loose vert when
               * it has only one face attached. So delete it in such
               * a case */
              BLI_assert(v2_e.get_verts());
              auto ov_index = std::get<0>(v2_e.get_verts().value());
              if (ov_index == v2_index) {
                ov_index = std::get<1>(v2_e.get_verts().value());
              }

              const auto ov_face_indices = this->get_checked_face_indices_of_vert(ov_index);
              if (ov_face_indices.size() == 1) {
                to_delete_vert_indices.add(ov_index);
              }
            }
          }
        }
      }

      for (const auto &face_index : to_delete_face_indices) {
        this->delink_face_edges(face_index);
        const auto f = this->delete_face(face_index);

        deleted_faces.append(f);
      }

      for (const auto &edge_index : to_delete_edge_indices) {
        const auto e = this->delete_edge(edge_index);

        deleted_edges.append(e);
      }

      for (const auto &vert_index : to_delete_vert_indices) {
        const auto v = this->delete_vert(vert_index);

        deleted_verts.append(v);
      }
    }

    /* delete the Node n1 */
    {
      auto &n1 = this->get_checked_node(n1_index);

      /* It is possible to have v1 which doesn't have a
       * corresponding v2 for this 3d edge but v1 should be entirely
       * removed if across seams is active, so make v1 as v2 by
       * making v1.node refer to n2 and removing the reference to v1
       * in n1 */
      if (across_seams) {
        const auto n1_verts = n1.get_verts();
        for (const auto &v1_index : n1_verts) {

          /* TODO(ish): might want to delete the faces and recreate
           * them so MeshDiff gets updated */

          auto &v1 = this->get_checked_vert(v1_index);
          v1.node = n2_index;
          auto &n2 = this->get_checked_node(n2_index);
          n2.verts.append(v1_index);

          n1.verts.remove_first_occurrence_and_reorder(v1_index);
        }
      }

      if (n1.get_verts().is_empty()) {
        const auto n1 = this->delete_node(n1_index);

        deleted_nodes.append(n1);
      }
      else {
        BLI_assert(across_seams == false);
      }
    }

    return MeshDiff(std::move(added_nodes),
                    std::move(added_verts),
                    std::move(added_edges),
                    std::move(added_faces),
                    std::move(deleted_nodes),
                    std::move(deleted_verts),
                    std::move(deleted_edges),
                    std::move(deleted_faces));
  }

  /**
   * Seam is the edge in UV space that has only one face.
   */
  bool is_vert_on_seam(const Vert<EVD> &vert) const
  {
    /* The vert is on a seam if any of it's adjacent edges is on a
     * seam */

    for (const auto &edge_index : vert.get_edges()) {
      if (this->is_edge_on_seam(edge_index)) {
        return true;
      }
    }

    return false;
  }

  /**
   * Easy call when only `vert_index` is available.
   */
  bool is_vert_on_seam(VertIndex vert_index) const
  {
    const auto &vert = this->get_checked_vert(vert_index);
    return is_vert_on_seam(vert);
  }

  /**
   * Boundary is the set of "3D edges" that have only a single
   * face. Not all meshes will have a boundary.
   */
  bool is_vert_on_boundary(const Vert<EVD> &vert) const
  {
    /* The vert is on a seam if any of it's adjacent edges is on a
     * boundary */

    /* TODO(ish): a simpler check might be to see
     * vert.get_edges().size() != vert.get_faces().size() */

    for (const auto &edge_index : vert.get_edges()) {
      if (this->is_edge_on_boundary(edge_index)) {
        return true;
      }
    }

    return false;
  }

  /**
   * Easy call when only `vert_index` is available.
   */
  bool is_vert_on_boundary(VertIndex vert_index) const
  {
    const auto &vert = this->get_checked_vert(vert_index);
    return is_vert_on_boundary(vert);
  }

  /**
   * Check both conditions at once
   */
  bool is_vert_on_seam_or_boundary(const Vert<EVD> &vert) const
  {
    return this->is_vert_on_seam(vert) || this->is_vert_on_boundary(vert);
  }

  /**
   * Easy call when only `vert_index` is available.
   */
  bool is_vert_on_seam_or_boundary(VertIndex vert) const
  {
    return this->is_vert_on_seam(vert) || this->is_vert_on_boundary(vert);
  }

  /**
   * An edge is loose when it doesn't have any faces.
   */
  bool is_edge_loose(const Edge<EED> &edge) const
  {
    return edge.get_faces().size() == 0;
  }

  /**
   * Easy call when only `edge_index` is available.
   */
  bool is_edge_loose(EdgeIndex edge_index) const
  {
    const auto &edge = this->get_checked_edge(edge_index);
    return is_edge_loose(edge);
  }

  /**
   * Seam is the edge in UV space that has only one face.
   *
   * Note: A loose edge cannot be considered to be on a seam.
   */
  bool is_edge_on_seam(const Edge<EED> &edge) const
  {
    return edge.get_faces().size() == 1;
  }

  /**
   * Easy call when only `edge_index` is available.
   */
  bool is_edge_on_seam(EdgeIndex edge_index) const
  {
    const auto &edge = this->get_checked_edge(edge_index);
    return is_edge_on_boundary(edge);
  }

  /**
   * Boundary is the set of "3D edges" that have only a single
   * face. Not all meshes will have a boundary.
   */
  bool is_edge_on_boundary(const Edge<EED> &edge) const
  {
    const auto [n1, n2] = this->get_checked_nodes_of_edge(edge, false);
    auto edge_indices = this->get_connecting_edge_indices(n1, n2);

    auto num_face = 0;
    for (const auto &edge_index : edge_indices) {
      const auto &e = this->get_checked_edge(edge_index);
      num_face += e.faces.size();
      if (num_face >= 1) {
        break;
      }
    }

    return num_face == 1;
  }

  /**
   * Easy call when only `edge_index` is available.
   */
  bool is_edge_on_boundary(EdgeIndex edge_index) const
  {
    const auto &edge = this->get_checked_edge(edge_index);
    return is_edge_on_boundary(edge);
  }

  /**
   * Easy call for checking all three conditions of the edge.
   */
  bool is_edge_loose_or_on_seam_or_boundary(const Edge<EED> &edge) const
  {
    return this->is_edge_on_seam(edge) || this->is_edge_on_boundary(edge) ||
           this->is_edge_loose(edge);
  }

  /**
   * Easy call when only `edge_index` is available.
   */
  bool is_edge_loose_or_on_seam_or_boundary(EdgeIndex edge_index) const
  {
    const auto &edge = this->get_checked_edge(edge_index);
    return is_edge_loose_or_on_seam_or_boundary(edge);
  }

  /**
   * An edge is flippable only if the edge has exactly 2 faces, and
   * both faces are triangulated.
   */
  bool is_edge_flippable(EdgeIndex edge_index, bool across_seams) const
  {
    /* Do more expensive test only if needed */
    if (across_seams) {
      const auto &edge = this->get_checked_edge(edge_index);
      if (is_edge_on_boundary(edge)) {
        return false;
      }
      const auto [n1, n2] = this->get_checked_nodes_of_edge(edge, false);
      auto edge_indices = this->get_connecting_edge_indices(n1, n2);

      auto num_faces = 0;
      for (const auto &edge_index : edge_indices) {
        const auto &e = this->get_checked_edge(edge_index);
        num_faces += e.faces.size();

        /* ensure triangulation */
        for (const auto &face_index : e.faces) {
          const auto &face = this->get_checked_face(face_index);
          if (face.verts.size() != 3) {
            return false;
          }
        }
      }
      /* Ensure only 2 faces exist for the "3D edge" */
      if (num_faces != 2) {
        return false;
      }
      return true;
    }

    const auto &edge = this->get_checked_edge(edge_index);
    /* Ensure only 2 faces exist for the edge */
    if (edge.faces.size() != 2) {
      return false;
    }

    /* Ensure triangulation */
    for (const auto &face_index : edge.faces) {
      const auto &face = this->get_checked_face(face_index);
      if (face.verts.size() != 3) {
        return false;
      }
    }

    /* Make sure there is no connecting edge between ov1 and ov2 */
    {
      const auto &f1 = this->get_checked_face(edge.faces[0]);
      const auto &f2 = this->get_checked_face(edge.faces[1]);
      const auto &ov1 = this->get_checked_other_vert(edge, f1);
      const auto &ov2 = this->get_checked_other_vert(edge, f2);

      if (this->get_connecting_edge_index(ov1.self_index, ov2.self_index)) {
        return false;
      }
    }

    return true;
  }

/**
 * Flips the edge specified and ensures triangulation of the Mesh.
 *
 * @param across_seams If true, think of edge as world space edge
 * and not UV space, this means, all the faces across all the edges
 * formed between the nodes of the given edge are used for flipping
 * regardless if it on a seam or not.
 *
 * Returns the `MeshDiff` that lead to the operation.
 *
 * Note, the caller must ensure the adjacent faces to the edge are
 * triangulated and that they are flippable using
 * `is_edge_flippable()`. In debug mode, it will assert, in release
 * mode, it is undefined behaviour.
 **/
#  ifndef NDEBUG
  /* TODO(ish): In debug mode, across seams is used. This is confusing
   * and bad code but prevents the warning for now, will fix later. */
  MeshDiff<END, EVD, EED, EFD> flip_edge_triangulate(EdgeIndex edge_index, bool across_seams)
#  else
  MeshDiff<END, EVD, EED, EFD> flip_edge_triangulate(EdgeIndex edge_index,
                                                     bool UNUSED(across_seams))
#  endif
  {
    BLI_assert(this->is_edge_flippable(edge_index, across_seams));

    /* This operation deletes the following:
     * when across_seams is true:
     *
     * when across_seams is false:
     * f1, f2, e
     *
     * This operation adds the following:
     * when across_seams is true:
     *
     * when across_seams is false:
     * 2 faces, 1 edge
     */

    /* Let `e` be the edge of `edge_index`
     * Let `v1` be the first vert of `e`
     * Let `v2` be the second vert of `e`
     * Let `f1` be the first face of `e`
     * Let `f2` be the second face of `e`
     * Let `ov1` be the other vert of `f1`
     * Let `ov2` be the other vert of `f2`
     */

    blender::Vector<NodeIndex> added_nodes;
    blender::Vector<VertIndex> added_verts;
    blender::Vector<EdgeIndex> added_edges;
    blender::Vector<FaceIndex> added_faces;
    blender::Vector<Node<END>> deleted_nodes;
    blender::Vector<Vert<EVD>> deleted_verts;
    blender::Vector<Edge<EED>> deleted_edges;
    blender::Vector<Face<EFD>> deleted_faces;

    auto &e = this->get_checked_edge(edge_index);
    BLI_assert(e.faces.size() == 1 || e.faces.size() == 2);
    if (e.faces.size() == 2) {
      auto f1_index = e.faces[0];
      auto f2_index = e.faces[1];

      this->delink_face_edges(f1_index);
      this->delink_face_edges(f2_index);

      auto f1 = this->delete_face(f1_index);
      auto f2 = this->delete_face(f2_index);

      auto e = this->delete_edge(edge_index);
      auto [v1, v2] = this->get_checked_verts_of_edge(e, false);
      auto v1_index = v1.self_index;
      auto v2_index = v2.self_index;

      auto &ov1 = this->get_checked_other_vert(e, f1);
      auto &ov2 = this->get_checked_other_vert(e, f2);
      auto ov1_index = ov1.self_index;
      auto ov2_index = ov2.self_index;

      /* Create the new edge only, `is_edge_flippable()` should have
       * already prevented the case of there being an edge between
       * ov1 and ov2 */
      auto &new_e = this->add_empty_edge();
      new_e.verts = {ov1_index, ov2_index};
      this->add_edge_ref_to_verts(new_e);
      added_edges.append(new_e.self_index);

      auto &new_f1 = this->add_empty_face(f1.normal);
      new_f1.verts = {v1_index, ov2_index, ov1_index};
      added_faces.append(new_f1.self_index);
      this->add_face_ref_to_edges(new_f1);
      BLI_assert(this->is_face_edges_linked(new_f1));

      auto &new_f2 = this->add_empty_face(f2.normal);
      new_f2.verts = {v2_index, ov1_index, ov2_index};
      added_faces.append(new_f2.self_index);
      this->add_face_ref_to_edges(new_f2);
      BLI_assert(this->is_face_edges_linked(new_f2));

      deleted_edges.append(std::move(e));
      deleted_faces.append(std::move(f1));
      deleted_faces.append(std::move(f2));
    }
    else {
      /* Do more expensive operation only if needed */
      auto [n1, n2] = this->get_checked_nodes_of_edge(e, false);
      auto edge_indices = this->get_connecting_edge_indices(n1, n2);
      BLI_assert(edge_indices.size() == 2);
      /* TODO(ish): implement this when it is necessary, edge flips
       * should actually be allowed only when the edges is not at a
       * seam or boundary */
    }

    BLI_assert(added_nodes.size() == 0);
    BLI_assert(added_verts.size() == 0);
    BLI_assert(added_edges.size() == 1);
    BLI_assert(added_faces.size() == 2);

    BLI_assert(deleted_nodes.size() == 0);
    BLI_assert(deleted_verts.size() == 0);
    BLI_assert(deleted_edges.size() == 1);
    BLI_assert(deleted_faces.size() == 2);
    return MeshDiff(std::move(added_nodes),
                    std::move(added_verts),
                    std::move(added_edges),
                    std::move(added_faces),
                    std::move(deleted_nodes),
                    std::move(deleted_verts),
                    std::move(deleted_edges),
                    std::move(deleted_faces));
  }

  float3 compute_face_normal(const Vert<EVD> &v1, const Vert<EVD> &v2, const Vert<EVD> &v3) const
  {
    const auto &n1 = this->get_checked_node_of_vert(v1);
    const auto &n2 = this->get_checked_node_of_vert(v2);
    const auto &n3 = this->get_checked_node_of_vert(v3);

    return float3::cross(n2.pos - n1.pos, n3.pos - n1.pos).normalized();
  }

  /**
   * Takes the first 3 verts to compute the face normal
   */
  void compute_face_normal(Face<EFD> &face)
  {
    BLI_assert(face.get_verts().size() >= 3);

    const auto &v1 = this->get_checked_vert(face.get_verts()[0]);
    const auto &v2 = this->get_checked_vert(face.get_verts()[1]);
    const auto &v3 = this->get_checked_vert(face.get_verts()[2]);

    face.normal = this->compute_face_normal(v1, v2, v3);
  }

  void compute_face_normal_all_faces()
  {
    for (auto &face : this->get_faces_mut()) {
      this->compute_face_normal(face);
    }
  }

  std::string serialize() const
  {
    std::stringstream ss;
    ss << "GenericMesh" << std::endl;

    msgpack::pack(ss, *this);

    return ss.str();
  }

 protected:
  /* all protected static methods */
  static constexpr inline float3 float3_unknown()
  {
    return float3(std::numeric_limits<float>::signaling_NaN());
  }

  static constexpr inline float2 float2_unknown()
  {
    return float2(std::numeric_limits<float>::signaling_NaN());
  }

  static inline bool float3_is_unknown(const float3 &f3)
  {
    return std::isnan(f3[0]) && std::isnan(f3[1]) && std::isnan(f3[2]);
  }

  static inline bool float2_is_unknown(const float2 &f2)
  {
    return std::isnan(f2[0]) && std::isnan(f2[1]);
  }

  /* all protected non-static methods */
  /**
   * Checks if `Node` with the given `node_index` exists in the mesh anymore.
   */
  inline bool does_node_exist(NodeIndex node_index) const
  {
    const auto op_node = this->nodes.get(node_index);
    if (op_node) {
      return true;
    }
    return false;
  }

  /**
   * Checks if `Vert` with the given `vert_index` exists in the mesh anymore.
   */
  inline bool does_vert_exist(VertIndex vert_index) const
  {
    const auto op_vert = this->verts.get(vert_index);
    if (op_vert) {
      return true;
    }
    return false;
  }

  /**
   * Checks if `Edge` with the given `edge_index` exists in the mesh anymore.
   */
  inline bool does_edge_exist(EdgeIndex edge_index) const
  {
    const auto op_edge = this->edges.get(edge_index);
    if (op_edge) {
      return true;
    }
    return false;
  }

  /**
   * Checks if `Face` with the given `face_index` exists in the mesh anymore.
   */
  inline bool does_face_exist(FaceIndex face_index) const
  {
    const auto op_face = this->faces.get(face_index);
    if (op_face) {
      return true;
    }
    return false;
  }

  /**
   * Get checked node
   */
  inline auto &get_checked_node(NodeIndex node_index)
  {
    auto op_node = this->nodes.get(node_index);
    BLI_assert(op_node);
    return op_node.value().get();
  }

  /**
   * Get const checked node
   */
  inline const auto &get_checked_node(NodeIndex node_index) const
  {
    auto op_node = this->nodes.get(node_index);
    BLI_assert(op_node);
    return op_node.value().get();
  }

  /**
   * Get checked vert
   */
  inline auto &get_checked_vert(VertIndex vert_index)
  {
    auto op_vert = this->verts.get(vert_index);
    BLI_assert(op_vert);
    return op_vert.value().get();
  }

  /**
   * Get const checked vert
   */
  inline const auto &get_checked_vert(VertIndex vert_index) const
  {
    auto op_vert = this->verts.get(vert_index);
    BLI_assert(op_vert);
    return op_vert.value().get();
  }

  /**
   * Get checked edge
   */
  inline auto &get_checked_edge(EdgeIndex edge_index)
  {
    auto op_edge = this->edges.get(edge_index);
    BLI_assert(op_edge);
    return op_edge.value().get();
  }

  /**
   * Get const checked edge
   */
  inline const auto &get_checked_edge(EdgeIndex edge_index) const
  {
    auto op_edge = this->edges.get(edge_index);
    BLI_assert(op_edge);
    return op_edge.value().get();
  }

  /**
   * Get checked face
   */
  inline auto &get_checked_face(FaceIndex face_index)
  {
    auto op_face = this->faces.get(face_index);
    BLI_assert(op_face);
    return op_face.value().get();
  }

  /**
   * Get const checked face
   */
  inline const auto &get_checked_face(FaceIndex face_index) const
  {
    auto op_face = this->faces.get(face_index);
    BLI_assert(op_face);
    return op_face.value().get();
  }

  blender::Set<FaceIndex> get_checked_face_indices_of_vert(const VertIndex vert_index) const
  {
    const auto &vert = this->get_checked_vert(vert_index);
    blender::Set<FaceIndex> face_indices;

    for (const auto &edge_index : vert.get_edges()) {
      const auto &edge = this->get_checked_edge(edge_index);

      for (const auto &face_index : edge.get_faces()) {
        face_indices.add(face_index);
      }
    }

    return face_indices;
  }

  /**
   * Gets the vert indices of the edge where v1.node is n1_index.
   *
   * Caller must ensure at least of one the verts of the edge must have
   * a reference to n1_index.
   */
  inline std::tuple<VertIndex, VertIndex> get_checked_vert_indices_of_edge_aligned_with_n1(
      const Edge<EED> &edge, const NodeIndex &n1_index) const
  {
    const auto [v1, v2] = this->get_checked_verts_of_edge(edge, false);
    auto v1_index = v1.get_self_index();
    auto v2_index = v2.get_self_index();
    /* Need to swap the verts if v1 does not point to n1 */
    if (v1.get_node().value() != n1_index) {
      std::swap(v1_index, v2_index);
    }
    BLI_assert(this->get_checked_vert(v1_index).get_node().value() == n1_index);
    return {v1_index, v2_index};
  }

  inline std::tuple<Vert<EVD> &, Vert<EVD> &> get_checked_verts_of_edge(const Edge<EED> &edge,
                                                                        bool verts_swapped)
  {
    BLI_assert(edge.verts);
    auto &edge_verts = edge.verts.value();

    if (verts_swapped) {
      auto &edge_vert_1 = this->get_checked_vert(std::get<1>(edge_verts));
      auto &edge_vert_2 = this->get_checked_vert(std::get<0>(edge_verts));
      return {edge_vert_1, edge_vert_2};
    }

    auto &edge_vert_1 = this->get_checked_vert(std::get<0>(edge_verts));
    auto &edge_vert_2 = this->get_checked_vert(std::get<1>(edge_verts));

    return {edge_vert_1, edge_vert_2};
  }

  inline std::tuple<const Vert<EVD> &, const Vert<EVD> &> get_checked_verts_of_edge(
      const Edge<EED> &edge, bool verts_swapped) const
  {
    BLI_assert(edge.verts);
    const auto &edge_verts = edge.verts.value();

    if (verts_swapped) {
      const auto &edge_vert_1 = this->get_checked_vert(std::get<1>(edge_verts));
      const auto &edge_vert_2 = this->get_checked_vert(std::get<0>(edge_verts));
      return {edge_vert_1, edge_vert_2};
    }

    const auto &edge_vert_1 = this->get_checked_vert(std::get<0>(edge_verts));
    const auto &edge_vert_2 = this->get_checked_vert(std::get<1>(edge_verts));

    return {edge_vert_1, edge_vert_2};
  }

  inline std::tuple<Node<END> &, Node<END> &> get_checked_nodes_of_edge(const Edge<EED> &edge,
                                                                        bool nodes_swapped)
  {
    auto [v1, v2] = this->get_checked_verts_of_edge(edge, nodes_swapped);

    BLI_assert(v1.node);
    BLI_assert(v2.node);
    auto &n1 = this->get_checked_node(v1.node.value());
    auto &n2 = this->get_checked_node(v2.node.value());

    return {n1, n2};
  }

  inline std::tuple<const Node<END> &, const Node<END> &> get_checked_nodes_of_edge(
      const Edge<EED> &edge, bool nodes_swapped) const
  {
    const auto [v1, v2] = this->get_checked_verts_of_edge(edge, nodes_swapped);

    BLI_assert(v1.node);
    BLI_assert(v2.node);
    const auto &n1 = this->get_checked_node(v1.node.value());
    const auto &n2 = this->get_checked_node(v2.node.value());

    return {n1, n2};
  }

  inline Node<END> &get_checked_node_of_vert(const Vert<EVD> &vert)
  {
    BLI_assert(vert.node);
    return this->get_checked_node(vert.node.value());
  }

  inline const Node<END> &get_checked_node_of_vert(const Vert<EVD> &vert) const
  {
    BLI_assert(vert.node);
    return this->get_checked_node(vert.node.value());
  }

  /**
   * Gives first vert index of a triangulated face that is not part of edge.
   *
   * Will abort in debug mode if face is not triangulated or edge is
   * not part of the face.
   * Will have undefined behaviour in release mode.
   **/
  inline Vert<EVD> &get_checked_other_vert(const Edge<EED> &edge, const Face<EFD> &face)
  {
    BLI_assert(face.verts.size() == 3);
    BLI_assert(face.has_edge(edge));

    const auto vert_1_index = face.verts[0];
    const auto vert_2_index = face.verts[1];
    const auto vert_3_index = face.verts[2];

    if (edge.has_vert(vert_1_index) == false) {
      return this->get_checked_vert(vert_1_index);
    }
    if (edge.has_vert(vert_2_index) == false) {
      return this->get_checked_vert(vert_2_index);
    }

    return this->get_checked_vert(vert_3_index);
  }

  /**
   * A const version of above
   */
  inline const Vert<EVD> &get_checked_other_vert(const Edge<EED> &edge,
                                                 const Face<EFD> &face) const
  {
    BLI_assert(face.verts.size() == 3);
    BLI_assert(face.has_edge(edge));

    const auto vert_1_index = face.verts[0];
    const auto vert_2_index = face.verts[1];
    const auto vert_3_index = face.verts[2];

    if (edge.has_vert(vert_1_index) == false) {
      return this->get_checked_vert(vert_1_index);
    }
    if (edge.has_vert(vert_2_index) == false) {
      return this->get_checked_vert(vert_2_index);
    }

    return this->get_checked_vert(vert_3_index);
  }

  /**
   * Get the edge indices of the `Face`
   */
  blender::Vector<EdgeIndex> get_edge_indices_of_face(const Face<EFD> &face) const
  {
    blender::Vector<EdgeIndex> edge_indices;
    auto vert_1_index = face.verts[0];
    auto vert_2_index = face.verts[0];
    for (auto i = 1; i <= face.verts.size(); i++) {
      vert_1_index = vert_2_index;
      if (i == face.verts.size()) {
        vert_2_index = face.verts[0];
      }
      else {
        vert_2_index = face.verts[i];
      }

      auto op_edge_index = this->get_connecting_edge_index(vert_1_index, vert_2_index);
      BLI_assert(op_edge_index);
      auto edge_index = op_edge_index.value();

      edge_indices.append(std::move(edge_index));
    }

    BLI_assert(edge_indices.size() == face.verts.size());

    return edge_indices;
  }

  /**
   * Computes extra data information for given node
   */
  void compute_info_node(Node<END> &UNUSED(node))
  {
  }

  /**
   * Computes extra data information for given vert
   */
  void compute_info_vert(Vert<EVD> &UNUSED(vert))
  {
  }

  /**
   * Computes extra data information for given edge
   */
  void compute_info_edge(Edge<EED> &UNUSED(edge))
  {
  }

  /**
   * Computes extra data information for given face
   */
  void compute_info_face(Face<EFD> &face)
  {
    this->compute_face_normal(face);
  }

  /**
   * For all added elements within mesh_diff, compute information
   * needed by the mesh. For example, face normals, etc.
   */
  void compute_info(const MeshDiff<END, EVD, EED, EFD> &mesh_diff)
  {
    for (const auto &node_index : mesh_diff.get_added_nodes()) {
      auto &node = this->get_checked_node(node_index);
      this->compute_info_node(node);
    }

    for (const auto &vert_index : mesh_diff.get_added_verts()) {
      auto &vert = this->get_checked_vert(vert_index);
      this->compute_info_vert(vert);
    }

    for (const auto &edge_index : mesh_diff.get_added_edges()) {
      auto &edge = this->get_checked_edge(edge_index);
      this->compute_info_edge(edge);
    }

    for (const auto &face_index : mesh_diff.get_added_faces()) {
      auto &face = this->get_checked_face(face_index);

      this->compute_info_face(face);
    }
  }

  Edge<EED> &add_checked_loose_edge(const VertIndex v1_index, const VertIndex v2_index)
  {
    /* Checks to ensure v1 and v2 are valid */
    {
      BLI_assert(this->has_vert(v1_index));
      BLI_assert(this->has_vert(v2_index));
      BLI_assert(this->get_connecting_edge_index(v1_index, v2_index) == std::nullopt);
    }

    auto &new_edge = this->add_empty_edge();
    new_edge.verts = std::make_tuple(v1_index, v2_index);
    this->add_edge_ref_to_verts(new_edge);

    return new_edge;
  }

 private:
  /* all private static methods */
  /* all private non-static methods */
  Node<END> &add_empty_node(float3 pos, float3 normal)
  {
    auto node_index = this->nodes.insert_with(
        [=](NodeIndex index) { return Node<END>(index, pos, normal); });

    return this->nodes.get(node_index).value().get();
  }

  Vert<EVD> &add_empty_vert(float2 uv)
  {
    auto vert_index = this->verts.insert_with(
        [=](VertIndex index) { return Vert<EVD>(index, uv); });

    return this->verts.get(vert_index).value().get();
  }

  Edge<EED> &add_empty_edge()
  {
    auto edge_index = this->edges.insert_with([=](EdgeIndex index) { return Edge<EED>(index); });

    return this->edges.get(edge_index).value().get();
  }

  Face<EFD> &add_empty_face(float3 normal)
  {
    auto face_index = this->faces.insert_with(
        [=](FaceIndex index) { return Face<EFD>(index, normal); });

    return this->faces.get(face_index).value().get();
  }

  Face<EFD> &add_face_triangulated(const VertIndex v1_index,
                                   const VertIndex v2_index,
                                   const VertIndex v3_index,
                                   float3 normal)
  {
    auto &f = this->add_empty_face(normal);

    f.verts.append(v1_index);
    f.verts.append(v2_index);
    f.verts.append(v3_index);

    this->add_face_ref_to_edges(f);

    return f;
  }

  /**
   * Adds an empty node with interpolation of the elements of `node_1`
   * and `node_2`.
   *
   * @return Reference to newly added node.
   */
  Node<END> &add_empty_interp_node(const Node<END> &node_1, const Node<END> &node_2)
  {
    auto pos = (node_1.pos + node_2.pos) * 0.5;
    /* The normal calculation might not be valid but good enough */
    auto normal = (node_1.normal + node_2.normal) * 0.5;

    std::optional<END> extra_data = std::nullopt;
    if (node_1.extra_data && node_2.extra_data) {
      extra_data = node_1.extra_data.value().interp(node_2.extra_data.value());
    }
    else if (node_1.extra_data) {
      extra_data = node_1.extra_data;
    }
    else if (node_2.extra_data) {
      extra_data = node_2.extra_data;
    }

    auto &new_node = this->add_empty_node(pos, normal);
    new_node.extra_data = extra_data;

    return new_node;
  }

  /**
   * Adds an empty vert with interpolation of the elements of `vert_1`
   * and `vert_2`.
   *
   * @return Reference to newly added vert.
   */
  Vert<EVD> &add_empty_interp_vert(const Vert<EVD> &vert_1, const Vert<EVD> &vert_2)
  {
    auto uv = (vert_1.uv + vert_2.uv) * 0.5;

    std::optional<EVD> extra_data = std::nullopt;
    if (vert_1.extra_data && vert_2.extra_data) {
      extra_data = vert_1.extra_data.value().interp(vert_2.extra_data.value());
    }
    else if (vert_1.extra_data) {
      extra_data = vert_1.extra_data;
    }
    else if (vert_2.extra_data) {
      extra_data = vert_2.extra_data;
    }

    auto &new_vert = this->add_empty_vert(uv);
    new_vert.extra_data = extra_data;

    return new_vert;
  }

  /**
   * Adds an empty edge with interpolation of the elements of `edge_1`
   * and `edge_2`.
   *
   * @return Reference to newly added edge.
   */
  Edge<EED> &add_empty_interp_edge(const Edge<EED> &edge_1, const Edge<EED> &edge_2)
  {
    std::optional<EED> extra_data = std::nullopt;
    if (edge_1.extra_data && edge_2.extra_data) {
      extra_data = edge_1.extra_data.value().interp(edge_2.extra_data.value());
    }
    else if (edge_1.extra_data) {
      extra_data = edge_1.extra_data;
    }
    else if (edge_2.extra_data) {
      extra_data = edge_2.extra_data;
    }

    auto &new_edge = this->add_empty_edge();
    new_edge.extra_data = extra_data;

    return new_edge;
  }

  /**
   * Adds an empty face with interpolation of the elements of `face_1`
   * and `face_2`.
   *
   * @return Reference to newly added face.
   */
  Face<EFD> &add_empty_interp_face(const Face<EFD> &face_1, const Face<EFD> &face_2)
  {
    /* The normal calculation might not be valid but good enough */
    auto normal = (face_1.normal + face_2.normal) * 0.5;

    std::optional<EFD> extra_data = std::nullopt;
    if (face_1.extra_data && face_2.extra_data) {
      extra_data = face_1.extra_data.value().interp(face_2.extra_data.value());
    }
    else if (face_1.extra_data) {
      extra_data = face_1.extra_data;
    }
    else if (face_2.extra_data) {
      extra_data = face_2.extra_data;
    }

    auto &new_face = this->add_empty_face(normal);
    new_face.extra_data = extra_data;

    return new_face;
  }

  /**
   * Adds the edge reference to the verts in `edge.verts`
   *
   * Caller should ensure edge.verts has been set
   *
   * note: this function should be called only once per edge
   */
  void add_edge_ref_to_verts(const Edge<EED> &edge)
  {
    BLI_assert(edge.verts);

    auto &v1 = this->get_checked_vert(std::get<0>(edge.verts.value()));
    BLI_assert(v1.edges.contains(edge.self_index) == false);
    v1.edges.append(edge.self_index);

    auto &v2 = this->get_checked_vert(std::get<1>(edge.verts.value()));
    BLI_assert(v2.edges.contains(edge.self_index) == false);
    v2.edges.append(edge.self_index);
  }

  /**
   * Adds the face reference to the edges created by `face.verts`
   *
   * Caller should ensure that `face.verts` has been set and edges
   * between the `face.verts` exist.
   *
   * note: this function should be called only once per face
   */
  void add_face_ref_to_edges(const Face<EFD> &face)
  {
    auto edge_indices = this->get_edge_indices_of_face(face);
    for (const auto &edge_index : edge_indices) {
      auto &edge = this->get_checked_edge(edge_index);

      BLI_assert(edge.faces.contains(face.self_index) == false);
      edge.faces.append(face.self_index);
    }
  }

  /**
   * Delete the node and update elements that refer to this node.
   *
   * This should always be preceeded with `delete_vert()` on all of
   * the `Node::verts` since a `Vert` without a `Node` is invalid.
   */
  Node<END> delete_node(NodeIndex node_index)
  {
    auto op_node = this->nodes.remove(node_index);
    BLI_assert(op_node);
    auto node = op_node.value();

    BLI_assert(node.verts.is_empty());

    return node;
  }

  /**
   * Delete the vert and update elements that refer to this vert.
   *
   * This should always be preceeded with `delete_edge()` on all of
   * the `Vert::edges` since a `Edge` without a both it's verts is
   * invalid.
   *
   * This may leave `Face`s that refer to this `Vert` invalid,
   * generally when the `Face` had 3 verts to begin with.
   *
   * This function does not (and cannot) remove the reference to this
   * `Vert` from the `Face`s that refer to it. `delete_edge()` should
   * have taken care of this.
   */
  Vert<EVD> delete_vert(VertIndex vert_index)
  {
    auto op_vert = this->verts.remove(vert_index);
    BLI_assert(op_vert);
    auto vert = op_vert.value();

    /* `Node` should exist, never can a `Vert` exist without the
     * `Node` but the other way round is possible
     *
     * Remove `Node`s reference to this vert */
    auto &node = this->get_checked_node_of_vert(vert);
    node.verts.remove_first_occurrence_and_reorder(vert.self_index);

    /* No `Edge` should be refering to this `Vert` */
    BLI_assert(vert.edges.is_empty());

    return vert;
  }

  /**
   * Delete the edge and update elements that refer to this edge.
   *
   * Also remove the edge verts from the face.
   *
   * The caller should always ensure that the face is valid even after
   * this operation.
   */
  Edge<EED> delete_edge(EdgeIndex edge_index)
  {
    auto op_edge = this->edges.remove(edge_index);
    BLI_assert(op_edge);
    auto edge = op_edge.value();

    /* Ensure the `Edge` has verts */
    BLI_assert(edge.verts);

    auto edge_vert_index_1 = std::get<0>(edge.verts.value());
    auto edge_vert_index_2 = std::get<1>(edge.verts.value());

    /* Remove the reference of this `Edge` from the edge verts */
    auto remove_edge_reference = [this, &edge](auto edge_vert_index) {
      auto &vert = this->get_checked_vert(edge_vert_index);
      vert.edges.remove_first_occurrence_and_reorder(edge.self_index);
    };
    remove_edge_reference(edge_vert_index_1);
    remove_edge_reference(edge_vert_index_2);

    /* Remove the edge.verts from the edge.faces
     *
     * If not done here, the link is never possible to remove unless
     * the Face itself is totally deleted.
     */
    for (const auto &face_index : edge.faces) {
      auto &face = this->get_checked_face(face_index);

      /* Some previous `delete_edge()` might have removed the edge
       * vert already from the face */
      auto maybe_remove_vert = [&face](auto &edge_vert_index) {
        auto v_pos = face.verts.first_index_of_try(edge_vert_index);
        if (v_pos != -1) {
          /* Need to do more expensive remove because the order of the
           * verts is important */
          face.verts.remove(v_pos);
        }
      };
      maybe_remove_vert(edge_vert_index_1);
      maybe_remove_vert(edge_vert_index_2);
    }

    return edge;
  }

  bool is_face_edges_linked(const Face<EFD> &face) const
  {
    if (face.verts.size() == 0) {
      /* No verts available, so no links possible */
      return false;
    }

    auto vert_1_index = face.verts[0];
    auto vert_2_index = face.verts[0];
    for (auto i = 1; i <= face.verts.size(); i++) {
      vert_1_index = vert_2_index;
      if (i == face.verts.size()) {
        vert_2_index = face.verts[0];
      }
      else {
        vert_2_index = face.verts[i];
      }

      auto op_edge_index = this->get_connecting_edge_index(vert_1_index, vert_2_index);
      /* TODO(ish): it might be possible to call this function once
       * the edges have been deleted, which can cause this assertion
       * to fail, need to figure out what the correct design decision
       * would be */
      BLI_assert(op_edge_index);
      auto edge_index = op_edge_index.value();
      auto &edge = this->get_checked_edge(edge_index);

      auto pos = edge.faces.first_index_of_try(face.self_index);
      /* this face should exist in the edge */
      if (pos == -1) {
        return false;
      }
    }

    return true;
  }

  /**
   * Remove this `Face`s from the `Edge`s formed by the `Vert`s of
   * this `Face`.
   *
   * This function is necessary when the `Face` must be deleted
   * when it shares `Edge`s with other `Face`s.
   */
  void delink_face_edges(FaceIndex face_index)
  {
    auto &face = this->get_checked_face(face_index);

    /* An earlier call to delete_edge and now this call can lead to
     * problems, so early exit if the verts were already removed from
     * the face. */
    if (face.verts.size() == 0) {
      return;
    }

    /* Would want to use `get_edges_of_face()` but that can lead to 2
     * loops, so duplicating that code here. (note: this needs to be
     * benchmarked to see if this duplication is necessary) */
    auto vert_1_index = face.verts[0];
    auto vert_2_index = face.verts[0];
    for (auto i = 1; i <= face.verts.size(); i++) {
      vert_1_index = vert_2_index;
      if (i == face.verts.size()) {
        vert_2_index = face.verts[0];
      }
      else {
        vert_2_index = face.verts[i];
      }

      auto op_edge_index = this->get_connecting_edge_index(vert_1_index, vert_2_index);
      BLI_assert(op_edge_index);
      auto edge_index = op_edge_index.value();
      auto &edge = this->get_checked_edge(edge_index);

      edge.faces.remove_first_occurrence_and_reorder(face.self_index);
    }
  }

  /**
   * Delete the face and update elements that refer to this face.
   *
   * This should always be preceeded with `delete_edge()` on all of
   * the `Edge`s that can be formed by the `Vert`s in the `Face` or
   * `delink_face_edges()` should be called once when other `Face`s
   * share `Edge`s with this `Face`.
   */
  Face<EFD> delete_face(FaceIndex face_index)
  {
    auto op_face = this->faces.remove(face_index);
    BLI_assert(op_face);
    auto face = op_face.value();

    BLI_assert(this->is_face_edges_linked(face) == false);

    return face;
  }
};

template<typename END, typename EVD, typename EED, typename EFD> class MeshDiff {
  blender::Vector<NodeIndex> added_nodes;
  blender::Vector<VertIndex> added_verts;
  blender::Vector<EdgeIndex> added_edges;
  blender::Vector<FaceIndex> added_faces;

  blender::Vector<Node<END>> deleted_nodes;
  blender::Vector<Vert<EVD>> deleted_verts;
  blender::Vector<Edge<EED>> deleted_edges;
  blender::Vector<Face<EFD>> deleted_faces;

 public:
  MeshDiff() = default;

  /* Move based constructor */
  MeshDiff(blender::Vector<NodeIndex> &&added_nodes,
           blender::Vector<VertIndex> &&added_verts,
           blender::Vector<EdgeIndex> &&added_edges,
           blender::Vector<FaceIndex> &&added_faces,

           blender::Vector<Node<END>> &&deleted_nodes,
           blender::Vector<Vert<EVD>> &&deleted_verts,
           blender::Vector<Edge<EED>> &&deleted_edges,
           blender::Vector<Face<EFD>> &&deleted_faces)
      : added_nodes(std::move(added_nodes)),
        added_verts(std::move(added_verts)),
        added_edges(std::move(added_edges)),
        added_faces(std::move(added_faces)),
        deleted_nodes(std::move(deleted_nodes)),
        deleted_verts(std::move(deleted_verts)),
        deleted_edges(std::move(deleted_edges)),
        deleted_faces(std::move(deleted_faces))
  {
  }

  void add_node(const NodeIndex &node_index)
  {
    this->added_nodes.append(node_index);
  }

  void add_vert(const VertIndex &vert_index)
  {
    this->added_verts.append(vert_index);
  }

  void add_edge(const EdgeIndex &edge_index)
  {
    this->added_edges.append(edge_index);
  }

  void add_face(const FaceIndex &face_index)
  {
    this->added_faces.append(face_index);
  }

  const auto &get_added_nodes() const
  {
    return this->added_nodes;
  }

  const auto &get_added_verts() const
  {
    return this->added_verts;
  }

  const auto &get_added_edges() const
  {
    return this->added_edges;
  }

  const auto &get_added_faces() const
  {
    return this->added_faces;
  }

  const auto &get_deleted_nodes() const
  {
    return this->deleted_nodes;
  }

  const auto &get_deleted_verts() const
  {
    return this->deleted_verts;
  }

  const auto &get_deleted_edges() const
  {
    return this->deleted_edges;
  }

  const auto &get_deleted_faces() const
  {
    return this->deleted_faces;
  }

  /**
   * Appends other into the current MeshDiff.
   *
   * It is generally a good idea to run
   * `remove_non_existing_elements()` after this function to ensure
   * correct `MeshDiff` does not contain invalid elements.
   */
  void append(const MeshDiff<END, EVD, EED, EFD> &other)
  {
    this->deleted_nodes.extend(other.deleted_nodes.as_span());
    this->deleted_verts.extend(other.deleted_verts.as_span());
    this->deleted_edges.extend(other.deleted_edges.as_span());
    this->deleted_faces.extend(other.deleted_faces.as_span());

    this->added_nodes.extend(other.added_nodes.as_span());
    this->added_verts.extend(other.added_verts.as_span());
    this->added_edges.extend(other.added_edges.as_span());
    this->added_faces.extend(other.added_faces.as_span());
  }

  /**
   * Removes elements from added elements that no longer exist in the
   * mesh.
   */
  void remove_non_existing_elements(const Mesh<END, EVD, EED, EFD> &mesh)
  {
    blender::Vector<NodeIndex> added_nodes;
    blender::Vector<VertIndex> added_verts;
    blender::Vector<EdgeIndex> added_edges;
    blender::Vector<FaceIndex> added_faces;

    for (const auto &node_index : this->added_nodes) {
      if (mesh.has_node(node_index)) {
        added_nodes.append(node_index);
      }
    }
    for (const auto &vert_index : this->added_verts) {
      if (mesh.has_vert(vert_index)) {
        added_verts.append(vert_index);
      }
    }
    for (const auto &edge_index : this->added_edges) {
      if (mesh.has_edge(edge_index)) {
        added_edges.append(edge_index);
      }
    }
    for (const auto &face_index : this->added_faces) {
      if (mesh.has_face(face_index)) {
        added_faces.append(face_index);
      }
    }

    this->added_nodes = added_nodes;
    this->added_verts = added_verts;
    this->added_edges = added_edges;
    this->added_faces = added_faces;
  }

  friend std::ostream &operator<<(std::ostream &stream,
                                  const MeshDiff<END, EVD, EED, EFD> &mesh_diff)
  {
    stream << "added_nodes: " << mesh_diff.get_added_nodes() << std::endl;
    stream << "added_verts: " << mesh_diff.get_added_verts() << std::endl;
    stream << "added_edges: " << mesh_diff.get_added_edges() << std::endl;
    stream << "added_faces: " << mesh_diff.get_added_faces() << std::endl;
    stream << "deleted_nodes: " << mesh_diff.get_deleted_nodes() << std::endl;
    stream << "deleted_verts: " << mesh_diff.get_deleted_verts() << std::endl;
    stream << "deleted_edges: " << mesh_diff.get_deleted_edges() << std::endl;
    stream << "deleted_faces: " << mesh_diff.get_deleted_faces() << std::endl;
    return stream;
  }
};

} /* namespace blender::bke::internal */

/* TODO(ish): Probably want to remove this later since it is mainly
 * for testing */
namespace blender::bke {

struct TempEmptyAdaptiveRemeshParams {
  float edge_length_min;
  float edge_length_max;
  float aspect_ratio_min;
  float change_in_vertex_normal_max;
  /* AdaptiveRemeshParamsFlags */
  uint32_t flags;
  /* AdaptiveRemeshParamsType */
  uint32_t type;
};

Mesh *__temp_empty_adaptive_remesh(const TempEmptyAdaptiveRemeshParams &params, Mesh *mesh);

} /* namespace blender::bke */

namespace msgpack {
MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{
  namespace adaptor {

  template<typename END, typename EVD, typename EED, typename EFD, typename Stream>
  void pack_mesh(msgpack::packer<Stream> &o,
                 const blender::bke::internal::Mesh<END, EVD, EED, EFD> &mesh)
  {
    blender::Map<blender::bke::internal::NodeIndex, uint64_t> node_pos_index_map;
    auto node_pos = 0;
    for (const auto &node : mesh.get_nodes()) {
      auto self_index = node.get_self_index();
      node_pos_index_map.add_new(self_index, node_pos);
      node_pos++;
    }

    blender::Map<blender::bke::internal::VertIndex, uint64_t> vert_pos_index_map;
    auto vert_pos = 0;
    for (const auto &vert : mesh.get_verts()) {
      auto self_index = vert.get_self_index();
      vert_pos_index_map.add_new(self_index, vert_pos);
      vert_pos++;
    }

    blender::Map<blender::bke::internal::EdgeIndex, uint64_t> edge_pos_index_map;
    auto edge_pos = 0;
    for (const auto &edge : mesh.get_edges()) {
      auto self_index = edge.get_self_index();
      edge_pos_index_map.add_new(self_index, edge_pos);
      edge_pos++;
    }

    blender::Map<blender::bke::internal::FaceIndex, uint64_t> face_pos_index_map;
    auto face_pos = 0;
    for (const auto &face : mesh.get_faces()) {
      auto self_index = face.get_self_index();
      face_pos_index_map.add_new(self_index, face_pos);
      face_pos++;
    }

    /* Need to store the arenas and the corresponding mappings
     * between the arena index and positional index of that element */
    o.pack_array(16);

    o.pack(std::string("nodes"));
    o.pack(mesh.get_nodes());
    o.pack(std::string("verts"));
    o.pack(mesh.get_verts());
    o.pack(std::string("edges"));
    o.pack(mesh.get_edges());
    o.pack(std::string("faces"));
    o.pack(mesh.get_faces());

    o.pack(std::string("node_map"));
    o.pack(node_pos_index_map);
    o.pack(std::string("vert_map"));
    o.pack(vert_pos_index_map);
    o.pack(std::string("edge_map"));
    o.pack(edge_pos_index_map);
    o.pack(std::string("face_map"));
    o.pack(face_pos_index_map);
  }

  template<> struct pack<blender::bke::internal::EmptyExtraData> {
    template<typename Stream>
    msgpack::packer<Stream> &operator()(msgpack::packer<Stream> &o,
                                        blender::bke::internal::EmptyExtraData /*unused*/) const
    {
      o.pack_array(1);

      o.pack("EmptyExtraData");

      return o;
    }
  };

  template<typename END, typename EVD, typename EED, typename EFD>
  struct pack<blender::bke::internal::Mesh<END, EVD, EED, EFD>> {
    template<typename Stream>
    msgpack::packer<Stream> &operator()(
        msgpack::packer<Stream> &o,
        const blender::bke::internal::Mesh<END, EVD, EED, EFD> &mesh) const
    {
      pack_mesh(o, mesh);

      return o;
    }
  };

  }  // namespace adaptor
}  // MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
}  // namespace msgpack

#endif /* __cplusplus */
