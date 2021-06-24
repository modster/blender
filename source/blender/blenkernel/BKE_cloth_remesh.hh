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

/******************************************************************************
 * reference http://graphics.berkeley.edu/papers/Narain-AAR-2012-11/index.html
 ******************************************************************************/

#include "BLI_assert.h"
#ifdef __cplusplus
extern "C" {
#endif

struct ClothModifierData;
struct Mesh;
struct Object;

void BKE_cloth_remesh(const struct Object *ob,
                      struct ClothModifierData *clmd,
                      struct Mesh *r_mesh);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

#  include <bits/stdint-uintn.h>
#  include <filesystem>
#  include <fstream>
#  include <iostream>
#  include <istream>
#  include <limits>
#  include <sstream>
#  include <string>
#  include <tuple>

#  include "BLI_float2.hh"
#  include "BLI_float3.hh"
#  include "BLI_generational_arena.hh"
#  include "BLI_vector.hh"

namespace blender::bke::internal {

template<typename> class Node;
template<typename> class Vert;
template<typename> class Edge;
template<typename> class Face;
template<typename, typename, typename, typename> class Mesh;
class MeshIO;

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

  bool has_vert(VertIndex vert_index)
  {
    if (this->verts) {
      if (std::get<0>(this->verts.value()) == vert_index ||
          std::get<1>(this->verts.value()) == vert_index) {
        return true;
      }
    }

    return false;
  }

  const auto &get_faces() const
  {
    return this->faces;
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
  enum FileType {
    FILETYPE_OBJ,
  };

  MeshIO() = default;

  bool read(const fs::path &filepath, FileType type)
  {
    if (type != FILETYPE_OBJ) {
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

  bool read(std::istream &&in, FileType type)
  {
    if (type == FILETYPE_OBJ) {
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

  bool write(const fs::path &filepath, FileType type)
  {
    if (type != FILETYPE_OBJ) {
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

  void write(std::ostream &out, FileType type)
  {
    if (type == FILETYPE_OBJ) {
      this->write_obj(out);
    }
    else {
      BLI_assert_unreachable();
    }
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
      else if (line.rfind("vn", 0) == 0) {
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
      out << "v " << normal.x << " " << normal.y << " " << normal.z << std::endl;
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

    /* TODO(ish): add line support */
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
                                                     VertIndex vert_2_index)
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

  void read(const MeshIO &reader)
  {
    const auto positions = reader.get_positions();
    const auto uvs = reader.get_uvs();
    const auto normals = reader.get_normals();
    const auto face_indices = reader.get_face_indices();
    const auto line_indices = reader.get_line_indices();

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
        node.verts.append(vert.self_index);

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

    /* TODO(ish): add support for lines */
  }

  void read_obj(const fs::path &filepath)
  {
    MeshIO reader;
    const auto reader_success = reader.read(filepath, MeshIO::FILETYPE_OBJ);
    BLI_assert(reader_success); /* must successfully load obj */

    this->read(reader);
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
};

} /* namespace blender::bke::internal */

#endif /* __cplusplus */
