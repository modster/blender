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

#include "DNA_meshdata_types.h"

#include "BLI_map.hh"
#include "BLI_vector.hh"

#include "IO_wavefront_obj.h"
#include "obj_export_mtl.hh"

namespace blender::io::obj {

/**
 * For an Object, total vertices/ UV vertices/ Normals written by previous objects
 * are added to its indices.
 */
struct IndexOffsets {
  int vertex_offset;
  int uv_vertex_offset;
  int normal_offset;
};

/**
 * Responsible for writing a .OBJ file.
 */
class OBJWriter {
 private:
  /**
   * Destination .OBJ file.
   */
  FILE *outfile_;
  const OBJExportParams &export_params_;

  IndexOffsets index_offsets_{0, 0, 0};
  /**
   * Total normals of an Object. It is not that same as `Mesh.tot_poly` due
   * to unknown smooth groups which add loop normals for smooth faces.
   *
   * Used for updating normal offset.
   */
  int per_object_tot_normals_ = 0;

 public:
  OBJWriter(const OBJExportParams &export_params) : export_params_(export_params)
  {
  }
  ~OBJWriter()
  {
    if (outfile_ && fclose(outfile_)) {
      std::cerr << "Error: could not close the OBJ file properly, file may be corrupted."
                << std::endl;
    }
  }

  bool init_writer(const char *filepath);

  void write_object_name(const OBJMesh &obj_mesh_data) const;
  void write_object_group(const OBJMesh &obj_mesh_data) const;
  void write_mtllib_name(const char *obj_filepath) const;
  void write_vertex_coords(const OBJMesh &obj_mesh_data) const;
  void write_uv_coords(OBJMesh &obj_mesh_data) const;
  void write_poly_normals(OBJMesh &obj_mesh_data) const;
  void write_smooth_group(const OBJMesh &obj_mesh_data,
                          int poly_index,
                          int &r_last_face_smooth_group) const;
  void write_poly_material(const OBJMesh &obj_mesh_data,
                           const int poly_index,
                           int16_t &r_last_face_mat_nr) const;
  void write_vertex_group(const OBJMesh &obj_mesh_data,
                          const int poly_index,
                          int16_t &r_last_face_vertex_group) const;
  void write_poly_elements(const OBJMesh &obj_mesh_data);
  void write_edges_indices(const OBJMesh &obj_mesh_data) const;
  void write_nurbs_curve(const OBJCurve &obj_nurbs_data) const;

  void update_index_offsets(const OBJMesh &obj_mesh_data);

 private:
  /* Based on export paramters, a writer function with correct syntax is needed. */
  typedef void (OBJWriter::*func_vert_uv_normal_indices)(Span<int>, Span<int>, Span<int>) const;

  func_vert_uv_normal_indices get_poly_element_writer(const OBJMesh &obj_mesh_data);
  void write_vert_uv_normal_indices(Span<int> vert_indices,
                                    Span<int> uv_indices,
                                    Span<int> normal_indices) const;
  void write_vert_normal_indices(Span<int> vert_indices,
                                 Span<int>,
                                 Span<int> normal_indices) const;
  void write_vert_uv_indices(Span<int> vert_indices, Span<int> uv_indices, Span<int>) const;
  void write_vert_indices(Span<int> vert_indices, Span<int>, Span<int>) const;
};

/**
 * Responsible for writing a .MTL file.
 */
class MTLWriter {
 private:
  char mtl_filepath_[FILE_MAX];
  FILE *mtl_outfile_;

 public:
  MTLWriter(const char *obj_filepath);
  ~MTLWriter();

  bool good() const;
  const char *mtl_file_path() const;
  void append_materials(const OBJMesh &mesh_to_export);

 private:
  void write_bsdf_properties(const MTLMaterial &mtl_material);
  void write_texture_map(const MTLMaterial &mtl_material,
                         const Map<const std::string, tex_map_XX>::Item &texture_map);
};
}  // namespace blender::io::obj
