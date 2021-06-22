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
class MeshReader;

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
};

template<typename END, typename EVD, typename EED, typename EFD> class Mesh {
  /* using declarations */
  /* static data members */
  /* non-static data members */
  ga::Arena<Node<END>> nodes;
  ga::Arena<Vert<EVD>> verts;
  ga::Arena<Edge<EED>> edges;
  ga::Arena<Face<EFD>> faces;

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

 protected:
  /* all protected static methods */
  /* all protected non-static methods */

 private:
  /* all private static methods */
  /* all private non-static methods */

  Node<END> &add_empty_node(float3 pos, float3 normal)
  {
    auto node_index = this->nodes.insert_with(
        [=](NodeIndex index) { return Node<END>(index, pos, normal); });

    return this->nodes.get(node_index);
  }

  Vert<EVD> &add_empty_vert(float2 uv)
  {
    auto vert_index = this->verts.insert_with(
        [=](VertIndex index) { return Vert<EVD>(index, uv); });

    return this->verts.get(vert_index);
  }

  Edge<EED> &add_empty_edge()
  {
    auto edge_index = this->edges.insert_with([=](EdgeIndex index) { return Edge<EED>(index); });

    return this->edges.get(edge_index);
  }

  Face<EFD> &add_empty_face(float3 normal)
  {
    auto face_index = this->faces.insert_with(
        [=](FaceIndex index) { return Face<EFD>(index, normal); });

    return this->faces.get(face_index);
  }
};

class MeshReader {
  using usize = uint64_t;
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

  MeshReader() = default;

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

    return true;
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

      if (line.rfind('v', 0) == 0) {
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
      else if (line.rfind("vt", 0) == 0) {
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
      else if (line.rfind("f", 0) == 0) {
        const auto line_toks = this->tokenize(line, ' ');

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
            face.append(std::make_tuple(pos_index - 1,
                                        std::numeric_limits<usize>::max(),
                                        std::numeric_limits<usize>::max()));
          }
          else if (indices_str.size() == 2) {
            std::istringstream isi_pos(indices_str[0]);
            std::istringstream isi_uv(indices_str[1]);
            usize pos_index;
            usize uv_index;
            isi_pos >> pos_index;
            isi_uv >> uv_index;
            face.append(
                std::make_tuple(pos_index - 1, uv_index - 1, std::numeric_limits<usize>::max()));
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

        this->face_indices.append(face);
      }
      else if (line.rfind("l", 0) == 0) {
        std::istringstream li(line);
        std::string temp;
        li >> temp;

        blender::Vector<usize> indices;
        usize index;
        while (li >> index) {
          indices.append(index - 1); /* obj starts from 1, we want to
                                      * start from 0 */
        }
        this->line_indices.append(indices);
      }
      else {
        /* unknown type, continuing */
        continue;
      }
    }

    return true;
  }
};

} /* namespace blender::bke::internal */

#endif /* __cplusplus */
