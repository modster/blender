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
 * The Original Code is Copyright (C) 2020 Blender Foundation.
 * All rights reserved.
 */

/** \file
 * \ingroup obj
 */

#pragma once

#include <fstream>

#include "IO_wavefront_obj.h"
#include "obj_import_mtl.hh"
#include "obj_import_objects.hh"

namespace blender::io::obj {

class OBJParser {
 private:
  const OBJImportParams &import_params_;
  std::ifstream obj_file_;
  Vector<std::string> mtl_libraries_;

 public:
  OBJParser(const OBJImportParams &import_params);

  void parse(Vector<std::unique_ptr<Geometry>> &r_all_geometries,
             GlobalVertices &r_global_vertices);
  Span<std::string> mtl_libraries() const;
};

class OBJStorer {
 private:
  Geometry &r_geom_;

 public:
  OBJStorer(Geometry &r_geom) : r_geom_(r_geom)
  {
  }
  void add_vertex(const StringRef rest_line, GlobalVertices &r_global_vertices);
  void add_vertex_normal(const StringRef rest_line, GlobalVertices &r_global_vertices);
  void add_uv_vertex(const StringRef rest_line, GlobalVertices &r_global_vertices);
  void add_edge(const StringRef rest_line,
                const VertexIndexOffset &offsets,
                GlobalVertices &r_global_vertices);
  void add_polygon(const StringRef rest_line,
                   const VertexIndexOffset &offsets,
                   const GlobalVertices &global_vertices,
                   const StringRef state_material_name,
                   const StringRef state_object_group,
                   const bool state_shaded_smooth);

  void set_curve_type(const StringRef rest_line,
                      const GlobalVertices &global_vertices,
                      const StringRef state_object_group,
                      VertexIndexOffset &r_offsets,
                      Vector<std::unique_ptr<Geometry>> &r_all_geometries);
  void set_curve_degree(const StringRef rest_line);
  void add_curve_vertex_indices(const StringRef rest_line, const GlobalVertices &global_vertices);
  void add_curve_parameters(const StringRef rest_line);

  void update_object_group(const StringRef rest_line, std::string &r_state_object_group) const;
  void update_polygon_material(const StringRef rest_line,
                               std::string &r_state_material_name) const;
  void update_smooth_group(const StringRef rest_line, bool &r_state_shaded_smooth) const;
};

enum class eOBJLineKey {
  V,
  VN,
  VT,
  F,
  L,
  CSTYPE,
  DEG,
  CURV,
  PARM,
  O,
  G,
  S,
  USEMTL,
  MTLLIB,
  COMMENT
};

constexpr eOBJLineKey line_key_str_to_enum(const std::string_view key_str)
{
  if (key_str == "v" || key_str == "V") {
    return eOBJLineKey::V;
  }
  if (key_str == "vn" || key_str == "VN") {
    return eOBJLineKey::VN;
  }
  if (key_str == "vt" || key_str == "VT") {
    return eOBJLineKey::VT;
  }
  if (key_str == "f" || key_str == "F") {
    return eOBJLineKey::F;
  }
  if (key_str == "l" || key_str == "L") {
    return eOBJLineKey::L;
  }
  if (key_str == "cstype" || key_str == "CSTYPE") {
    return eOBJLineKey::CSTYPE;
  }
  if (key_str == "deg" || key_str == "DEG") {
    return eOBJLineKey::DEG;
  }
  if (key_str == "curv" || key_str == "CURV") {
    return eOBJLineKey::CURV;
  }
  if (key_str == "parm" || key_str == "PARM") {
    return eOBJLineKey::PARM;
  }
  if (key_str == "o" || key_str == "O") {
    return eOBJLineKey::O;
  }
  if (key_str == "g" || key_str == "G") {
    return eOBJLineKey::G;
  }
  if (key_str == "s" || key_str == "S") {
    return eOBJLineKey::S;
  }
  if (key_str == "usemtl" || key_str == "USEMTL") {
    return eOBJLineKey::USEMTL;
  }
  if (key_str == "mtllib" || key_str == "MTLLIB") {
    return eOBJLineKey::MTLLIB;
  }
  if (key_str == "#") {
    return eOBJLineKey::COMMENT;
  }
  return eOBJLineKey::COMMENT;
}

/**
 * All texture map options with number of arguments they accept.
 */
class TextureMapOptions {
 private:
  Map<const std::string, int> tex_map_options;

 public:
  TextureMapOptions()
  {
    tex_map_options.add_new("-blendu", 1);
    tex_map_options.add_new("-blendv", 1);
    tex_map_options.add_new("-boost", 1);
    tex_map_options.add_new("-mm", 2);
    tex_map_options.add_new("-o", 3);
    tex_map_options.add_new("-s", 3);
    tex_map_options.add_new("-t", 3);
    tex_map_options.add_new("-texres", 1);
    tex_map_options.add_new("-clamp", 1);
    tex_map_options.add_new("-bm", 1);
    tex_map_options.add_new("-imfchan", 1);
  }

  /**
   * All valid option strings.
   */
  Map<const std::string, int>::KeyIterator all_options() const
  {
    return tex_map_options.keys();
  }

  int number_of_args(StringRef option) const
  {
    return tex_map_options.lookup_as(std::string(option));
  }
};

class MTLParser {
 private:
  char mtl_file_path_[FILE_MAX];
  /**
   * Directory in which the MTL file is found.
   */
  char mtl_dir_path_[FILE_MAX];
  std::ifstream mtl_file_;

 public:
  MTLParser(StringRef mtl_library_, StringRefNull obj_filepath);

  void parse_and_store(Map<std::string, std::unique_ptr<MTLMaterial>> &r_mtl_materials);
};
}  // namespace blender::io::obj
