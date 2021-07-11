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

/**********************************************************************
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
 * ********************************************************************/

#include "BKE_mesh.h"
#include "BLI_assert.h"

#include "BKE_customdata.h"

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ClothModifierData;
struct Object;

Mesh *BKE_cloth_remesh(struct Object *ob, struct ClothModifierData *clmd, struct Mesh *mesh);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

#  include <bits/stdint-uintn.h>
#  include <cmath>
#  include <filesystem>
#  include <fstream>
#  include <iostream>
#  include <istream>
#  include <limits>
#  include <optional>
#  include <sstream>
#  include <string>
#  include <tuple>

#  include "BLI_float2.hh"
#  include "BLI_float3.hh"
#  include "BLI_generational_arena.hh"
#  include "BLI_map.hh"
#  include "BLI_vector.hh"

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

  void set_extra_data(T extra_data)
  {
    this->extra_data = extra_data;
  }

  const auto &get_verts() const
  {
    return this->verts;
  }

  friend std::ostream &operator<<(std::ostream &stream, const Node &node)
  {
    stream << "(self_index: " << node.self_index << ", verts: " << node.verts
           << ", pos: " << node.pos << ", normal: " << node.normal << ")";
    return stream;
  }

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

  void set_extra_data(T extra_data)
  {
    this->extra_data = extra_data;
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

  void set_extra_data(T extra_data)
  {
    this->extra_data = extra_data;
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

  friend std::ostream &operator<<(std::ostream &stream, const Edge &edge)
  {
    stream << "(self_index: " << edge.self_index << ", faces: " << edge.faces
           << ", verts: " << edge.verts << ")";
    return stream;
  }

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

  void set_extra_data(T extra_data)
  {
    this->extra_data = extra_data;
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

  /* all public static methods */
  /* all public non-static methods */
  const auto &get_nodes() const
  {
    return this->nodes;
  }

  const auto &get_verts() const
  {
    return this->verts;
  }

  const auto &get_edges() const
  {
    return this->edges;
  }

  const auto &get_faces() const
  {
    return this->faces;
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
   * Gives first vert index of face that is not part of edge.
   * This should be called only when the face has 3 verts, will return
   * `std::nullopt` otherwise.
   **/
  inline std::optional<VertIndex> get_other_vert_index(EdgeIndex edge_index, FaceIndex face_index)
  {

    auto op_face = this->faces.get(face_index);
    BLI_assert(op_face);
    auto &face = op_face.value().get();

    if (face.verts.size() != 3) {
      return std::nullopt;
    }

    auto [vert_1_index, vert_2_index, vert_3_index] = face.verts;

    auto op_edge = this->edges.get(edge_index);
    BLI_assert(op_edge);
    auto &edge = op_edge.value().get();

    if (edge.has_vert(vert_1_index) == false) {
      return vert_1_index;
    }
    if (edge.has_vert(vert_2_index) == false) {
      return vert_2_index;
    }
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

    std::cout << "line_indices: " << line_indices << std::endl;

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
   * Returns the `MeshDiff` that lead to the operation.
   *
   * Note, the caller must ensure the adjacent faces to the edge are
   * triangulated. In debug mode, it will assert, in release mode, it
   * is undefined behaviour.
   **/
  MeshDiff<END, EVD, EED, EFD> split_edge_triangulate(EdgeIndex edge_index, bool across_seams)
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

    std::cout << "edge_indices: " << edge_indices << std::endl;

    /* Create the new new by interpolating the nodes of the edge */
    auto &new_node = this->add_empty_interp_node(edge_node_1, edge_node_2);

    for (const auto &edge_index : edge_indices) {
      auto &edge_a = this->get_checked_edge(edge_index);
      auto [edge_vert_1_a, edge_vert_2_a] = this->get_checked_verts_of_edge(edge_a, false);

      /* Create the new vert by interpolating the verts of the edge */
      auto &new_vert = this->add_empty_interp_vert(edge_vert_1_a, edge_vert_2_a);

      auto [edge_vert_1_b, edge_vert_2_b] = this->get_checked_verts_of_edge(edge_a, false);

      /* Link new_vert with new_node */
      new_vert.node = new_node.self_index;
      new_node.verts.append(new_vert.self_index);

      /* Create edges between edge_vert_1, new_vert, edge_vert_2 */
      auto &new_edge_1 = this->add_empty_edge();
      new_edge_1.verts = {edge_vert_1_b.self_index, new_vert.self_index};
      added_edges.append(new_edge_1.self_index);
      auto new_edge_1_index = new_edge_1.self_index;

      auto &new_edge_2 = this->add_empty_edge();
      new_edge_2.verts = {new_vert.self_index, edge_vert_2_b.self_index};
      added_edges.append(new_edge_2.self_index);
      auto new_edge_2_index = new_edge_2.self_index;

      /* Need to reinitialize edge because `add_empty_edge()` may have
       * reallocated `this->edges` */
      auto &edge_b = this->get_checked_edge(edge_index);
      auto faces = edge_b.faces;

      for (const auto &face_index : faces) {
        this->delink_face_edges(face_index);
        auto face = this->delete_face(face_index);

        /* Ensure the faces are triangulated before calling this function */
        BLI_assert(face.verts.size() == 3);

        auto &other_vert = this->get_checked_other_vert(edge_b, face);

        /* TODO(ish): Ordering of the verts and nodes needs to be found correctly */
        /* Handle new face and new edge creation */
        {
          /* Handle new edge creation between new_vert and other_vert */
          auto &new_edge = this->add_empty_edge();
          new_edge.verts = std::make_tuple(other_vert.self_index, new_vert.self_index);
          added_edges.append(new_edge.self_index);

          auto &new_face_1 = this->add_empty_face(face.normal);
          new_face_1.verts.append(edge_vert_1_b.self_index);
          new_face_1.verts.append(other_vert.self_index);
          new_face_1.verts.append(new_vert.self_index);
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
          new_face_2.verts.append(other_vert.self_index);
          new_face_2.verts.append(edge_vert_2_b.self_index);
          new_face_2.verts.append(new_vert.self_index);
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
   **/
  MeshDiff<END, EVD, EED, EFD> collapse_edge_triangulate(EdgeIndex edge_index,
                                                         bool verts_swapped,
                                                         bool across_seams)
  {
    /* This operation will delete the following-
     * the edge specified, faces incident to the edge. v2, n2 if
     * verts_swapped is true. v1, n1 if verts_swapped is false. One of the
     * edge per face (edge vert and other vert).
     *
     * This operation will add the following-
     * None
     */

    blender::Vector<NodeIndex> added_nodes;
    blender::Vector<VertIndex> added_verts;
    blender::Vector<EdgeIndex> added_edges;
    blender::Vector<FaceIndex> added_faces;
    blender::Vector<Node<END>> deleted_nodes;
    blender::Vector<Vert<EVD>> deleted_verts;
    blender::Vector<Edge<EED>> deleted_edges;
    blender::Vector<Face<EFD>> deleted_faces;

    auto &edge_a = this->get_checked_edge(edge_index);
    auto [v1, v2] = this->get_checked_verts_of_edge(edge_a, verts_swapped);
    auto v1_index = v1.self_index;
    auto n1_index = v1.node.value();

    /* This point forward, the vert to be removed is v1, v2 continues
     * to exist */

    auto faces = edge_a.faces;

    for (const auto &face_index : faces) {
      auto face = this->delete_face(face_index);

      auto &edge_b = this->get_checked_edge(edge_index);

      auto &other_vert = this->get_checked_other_vert(edge_b, face);

      /* delete edge between v1 and other_vert */
      {
        auto op_e_index = this->get_connecting_edge_index(v1_index, other_vert.self_index);
        BLI_assert(op_e_index);
        auto e = this->delete_edge(op_e_index.value());

        deleted_edges.append(std::move(e));
      }

      deleted_faces.append(std::move(face));
    }

    /* for each edge of v1, change that edge's verts to (vx, v2) */
    for (const auto &e_index : v1.edges) {
      Edge<EED> &e = this->get_checked_edge(e_index);

      /* we don't want to mess with edge between v1 and v2 */
      if (e.has_vert(v2.self_index)) {
        continue;
      }

      BLI_assert(e.verts);
      auto &verts = e.verts.value();
      if (std::get<0>(verts) == v1_index) {
        e.verts = {std::get<1>(verts), v2.self_index};
      }
      else {
        e.verts = {std::get<0>(verts), v2.self_index};
      }
    }

    /* delete edge */
    {
      auto edge = this->delete_edge(edge_index);
      deleted_edges.append(std::move(edge));
    }

    /* delete v1 */
    {
      auto v1 = this->delete_vert(v1_index);
      deleted_verts.append(std::move(v1));
    }

    /* delete n1 */
    {
      auto n1 = this->delete_node(n1_index);
      deleted_nodes.append(std::move(n1));
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

 protected:
  /* all protected static methods */
  /* all protected non-static methods */

 private:
  /* all private static methods */
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

  inline Node<END> &get_checked_node_of_vert(const Vert<EVD> &vert)
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
};

} /* namespace blender::bke::internal */

#endif /* __cplusplus */
