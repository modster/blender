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

#include <cstdio>

#include "BKE_blender_version.h"

#include "obj_export_file_writer.hh"
#include "obj_export_mesh.hh"
#include "obj_export_mtl.hh"
#include "obj_export_nurbs.hh"

namespace blender::io::obj {

/* Default values of some parameters. */
const int SMOOTH_GROUP_DISABLED = 0;
const int SMOOTH_GROUP_DEFAULT = 1;

/**
 * Write one line of polygon indices as f v1/vt1/vn1 v2/vt2/vn2 ... .
 */
void OBJWriter::write_vert_uv_normal_indices(Span<uint> vert_indices,
                                             Span<uint> uv_indices,
                                             Span<uint> normal_indices,
                                             const uint tot_loop) const
{
  fprintf(outfile_, "f");
  for (uint j = 0; j < tot_loop; j++) {
    fprintf(outfile_,
            " %u/%u/%u",
            vert_indices[j] + index_offsets_.vertex_offset + 1,
            uv_indices[j] + index_offsets_.uv_vertex_offset + 1,
            normal_indices[j] + index_offsets_.normal_offset + 1);
  }
  fprintf(outfile_, "\n");
}

/**
 * Write one line of polygon indices as f v1//vn1 v2//vn2 ... .
 */
void OBJWriter::write_vert_normal_indices(Span<uint> vert_indices,
                                          Span<uint>,
                                          Span<uint> normal_indices,
                                          const uint tot_loop) const
{
  fprintf(outfile_, "f");
  for (uint j = 0; j < tot_loop; j++) {
    fprintf(outfile_,
            " %u//%u",
            vert_indices[j] + index_offsets_.vertex_offset + 1,
            normal_indices[j] + index_offsets_.normal_offset + 1);
  }
  fprintf(outfile_, "\n");
}

/**
 * Write one line of polygon indices as f v1/vt1 v2/vt2 ... .
 */
void OBJWriter::write_vert_uv_indices(Span<uint> vert_indices,
                                      Span<uint> uv_indices,
                                      Span<uint>,
                                      const uint tot_loop) const
{
  fprintf(outfile_, "f");
  for (uint j = 0; j < tot_loop; j++) {
    fprintf(outfile_,
            " %u/%u",
            vert_indices[j] + index_offsets_.vertex_offset + 1,
            uv_indices[j] + index_offsets_.uv_vertex_offset + 1);
  }
  fprintf(outfile_, "\n");
}

/**
 *  Write one line of polygon indices as f v1 v2 ... .
 */
void OBJWriter::write_vert_indices(Span<uint> vert_indices,
                                   Span<uint>,
                                   Span<uint>,
                                   const uint tot_loop) const
{
  fprintf(outfile_, "f");
  for (uint j = 0; j < tot_loop; j++) {
    fprintf(outfile_, " %u", vert_indices[j] + index_offsets_.vertex_offset + 1);
  }
  fprintf(outfile_, "\n");
}

/**
 * Open the OBJ file and write file header.
 * \return Whether the destination file is writable.
 */
bool OBJWriter::init_writer(const char *filepath)
{
  outfile_ = fopen(filepath, "w");
  if (!outfile_) {
    std::perror(std::string("Error in creating the file at: ").append(filepath).c_str());
    return false;
  }
  fprintf(outfile_, "# Blender %s\n# www.blender.org\n", BKE_blender_version_string());
  return true;
}

/**
 * Write file name of Material Library in OBJ file.
 */
void OBJWriter::write_mtllib_name(const char *mtl_filepath) const
{
  /* Split MTL file path into parent directory and filename. */
  char mtl_file_name[FILE_MAXFILE];
  char mtl_dir_name[FILE_MAXDIR];
  BLI_split_dirfile(mtl_filepath, mtl_dir_name, mtl_file_name, FILE_MAXDIR, FILE_MAXFILE);
  fprintf(outfile_, "mtllib %s\n", mtl_file_name);
}

/**
 * Write object name conditionally with mesh and material name.
 */
void OBJWriter::write_object_name(const OBJMesh &obj_mesh_data) const
{
  const char *object_name = obj_mesh_data.get_object_name();

  if (!export_params_.export_object_groups) {
    fprintf(outfile_, "o %s\n", object_name);
  }
  else {
    const char *object_mesh_name = obj_mesh_data.get_object_mesh_name();
    if (export_params_.export_materials && export_params_.export_material_groups) {
      const char *object_material_name = obj_mesh_data.get_object_material_name(0);
      fprintf(outfile_, "g %s_%s_%s\n", object_name, object_mesh_name, object_material_name);
    }
    else {
      fprintf(outfile_, "g %s_%s\n", object_name, object_mesh_name);
    }
  }
}

/**
 * Write vertex coordinates for all vertices as v x y z .
 */
void OBJWriter::write_vertex_coords(const OBJMesh &obj_mesh_data) const
{
  const int tot_vertices = obj_mesh_data.tot_vertices();
  for (uint i = 0; i < tot_vertices; i++) {
    float3 vertex = obj_mesh_data.calc_vertex_coords(i, export_params_.scaling_factor);
    fprintf(outfile_, "v %f %f %f\n", vertex[0], vertex[1], vertex[2]);
  }
}

/**
 * Write UV vertex coordinates for all vertices as vt u v .
 * \note UV indices are stored here, but written later.
 */
void OBJWriter::write_uv_coords(OBJMesh &obj_mesh_data) const
{
  Vector<std::array<float, 2>> uv_coords;
  /* UV indices are calculated and stored in an OBJMesh member here. */
  obj_mesh_data.store_uv_coords_and_indices(uv_coords);

  for (const std::array<float, 2> &uv_vertex : uv_coords) {
    fprintf(outfile_, "vt %f %f\n", uv_vertex[0], uv_vertex[1]);
  }
}

/**
 * Write loop normals for smooth-shaded polygons, and face normals otherwise, as vn x y z .
 */
void OBJWriter::write_poly_normals(OBJMesh &obj_mesh_data) const
{
  obj_mesh_data.ensure_mesh_normals();
  Vector<float3> lnormals;
  const int tot_polygons = obj_mesh_data.tot_polygons();
  for (uint i = 0; i < tot_polygons; i++) {
    if (obj_mesh_data.is_ith_poly_smooth(i)) {
      obj_mesh_data.calc_loop_normals(i, lnormals);
      for (const float3 &lnormal : lnormals) {
        fprintf(outfile_, "vn %f %f %f\n", lnormal[0], lnormal[1], lnormal[2]);
      }
    }
    else {
      float3 poly_normal = obj_mesh_data.calc_poly_normal(i);
      fprintf(outfile_, "vn %f %f %f\n", poly_normal[0], poly_normal[1], poly_normal[2]);
    }
  }
}

/**
 * Write smooth group if the polygon at given index is shaded smooth and export settings specify
 * so. If the polygon is not shaded smooth, write "0".
 */
void OBJWriter::write_smooth_group(const OBJMesh &obj_mesh_data,
                                   const uint poly_index,
                                   int &r_last_face_smooth_group) const
{
  int current_group = SMOOTH_GROUP_DISABLED;
  if (!export_params_.export_smooth_groups && obj_mesh_data.is_ith_poly_smooth(poly_index)) {
    /* Smooth group calculation is disabled, but face is smooth. */
    current_group = SMOOTH_GROUP_DEFAULT;
  }
  else if (obj_mesh_data.is_ith_poly_smooth(poly_index)) {
    /* Smooth group calc enabled and face is smooth and so find the group. */
    current_group = obj_mesh_data.ith_smooth_group(poly_index);
  }

  if (current_group == r_last_face_smooth_group) {
    /* Group has already been written, even if it is "s 0". */
    return;
  }
  fprintf(outfile_, "s %d\n", current_group);
  r_last_face_smooth_group = current_group;
}

/**
 * Write material name and material group of a face in the OBJ file.
 * \note It doesn't write to the material library.
 */
void OBJWriter::write_poly_material(const OBJMesh &obj_mesh_data,
                                    const uint poly_index,
                                    short &r_last_face_mat_nr) const
{
  if (!export_params_.export_materials || obj_mesh_data.tot_materials() <= 0) {
    return;
  }
  const short curr_mat_nr = obj_mesh_data.ith_poly_matnr(poly_index);
  /* Whenever a face with a new material is encountered, write its material and/or group, otherwise
   * pass. */
  if (r_last_face_mat_nr == curr_mat_nr) {
    return;
  }
  const char *mat_name = obj_mesh_data.get_object_material_name(curr_mat_nr);
  if (export_params_.export_material_groups) {
    const char *object_name = obj_mesh_data.get_object_name();
    const char *object_mesh_name = obj_mesh_data.get_object_mesh_name();
    fprintf(outfile_, "g %s_%s_%s\n", object_name, object_mesh_name, mat_name);
  }
  fprintf(outfile_, "usemtl %s\n", mat_name);
  r_last_face_mat_nr = curr_mat_nr;
}

/**
 * Write the name of the deform group of a polygon. If no vertex group is found in
 * the polygon, "off" is written.
 */
void OBJWriter::write_vertex_group(const OBJMesh &obj_mesh_data,
                                   const uint poly_index,
                                   short &r_last_poly_vertex_group) const
{
  if (!export_params_.export_vertex_groups) {
    return;
  }
  const short current_group = obj_mesh_data.get_poly_deform_group_index(poly_index);

  if (current_group == r_last_poly_vertex_group) {
    /* No vertex group found in this face, just like in the last iteration. */
    return;
  }
  r_last_poly_vertex_group = current_group;
  if (current_group == NOT_FOUND) {
    fprintf(outfile_, "g off\n");
    return;
  }
  fprintf(outfile_, "g %s\n", obj_mesh_data.get_poly_deform_group_name(current_group));
}

/**
 * Select which syntax to write polygon elements with.
 */
OBJWriter::func_vert_uv_normal_indices OBJWriter::get_poly_element_writer(
    const OBJMesh &obj_mesh_data)
{
  if (export_params_.export_normals) {
    if (export_params_.export_uv && (obj_mesh_data.tot_uv_vertices() > 0)) {
      /* Write both normals and UV indices. */
      return &OBJWriter::write_vert_uv_normal_indices;
    }
    /* Write normals indices. */
    return &OBJWriter::write_vert_normal_indices;
  }
  /* Write UV indices. */
  if (export_params_.export_uv && (obj_mesh_data.tot_uv_vertices() > 0)) {
    return &OBJWriter::write_vert_uv_indices;
  }
  /* Write neither normals nor UV indices. */
  return &OBJWriter::write_vert_indices;
}

/**
 * Define and write face elements with at least vertex indices, and conditionally with UV vertex
 * indices and face normal indices. Also write groups: smooth, vertex, material.
 *  \note UV indices are stored while writing UV vertices.
 */
void OBJWriter::write_poly_elements(const OBJMesh &obj_mesh_data)
{
  int last_face_smooth_group = NEGATIVE_INIT;
  short last_face_vertex_group = NEGATIVE_INIT;
  short last_face_mat_nr = NEGATIVE_INIT;

  func_vert_uv_normal_indices poly_element_writer = get_poly_element_writer(obj_mesh_data);

  Vector<uint> face_vertex_indices;
  Vector<uint> face_normal_indices;
  /* Reset for every Object. */
  per_object_tot_normals_ = 0;
  const int tot_polygons = obj_mesh_data.tot_polygons();
  for (uint i = 0; i < tot_polygons; i++) {
    const int totloop = obj_mesh_data.ith_poly_totloop(i);
    obj_mesh_data.calc_poly_vertex_indices(i, face_vertex_indices);
    /* For an Object, a normal index depends on how many have been written before it.
     * This is unknown because of smooth shading. So pass "per object total normals"
     * and update it after each call. */
    per_object_tot_normals_ += obj_mesh_data.calc_poly_normal_indices(
        i, per_object_tot_normals_, face_normal_indices);

    write_smooth_group(obj_mesh_data, i, last_face_smooth_group);
    write_vertex_group(obj_mesh_data, i, last_face_vertex_group);
    write_poly_material(obj_mesh_data, i, last_face_mat_nr);
    (this->*poly_element_writer)(
        face_vertex_indices, obj_mesh_data.uv_indices(i), face_normal_indices, totloop);
  }
}

/**
 * Write loose edges of a mesh as l v1 v2 .
 */
void OBJWriter::write_edges_indices(const OBJMesh &obj_mesh_data) const
{
  obj_mesh_data.ensure_mesh_edges();
  const int tot_edges = obj_mesh_data.tot_edges();
  for (uint edge_index = 0; edge_index < tot_edges; edge_index++) {
    std::optional<std::array<int, 2>> vertex_indices = obj_mesh_data.calc_loose_edge_vert_indices(
        edge_index);
    if (!vertex_indices) {
      continue;
    }
    fprintf(outfile_,
            "l %u %u\n",
            (*vertex_indices)[0] + index_offsets_.vertex_offset + 1,
            (*vertex_indices)[1] + index_offsets_.vertex_offset + 1);
  }
}

/**
 * Write a NURBS curve to the OBJ file in parameter form.
 */
void OBJWriter::write_nurbs_curve(const OBJCurve &obj_nurbs_data) const
{
  const int tot_nurbs = obj_nurbs_data.tot_nurbs();
  for (int i = 0; i < tot_nurbs; i++) {
    /* Total control points in a nurbs. */
    const int tot_points = obj_nurbs_data.get_nurbs_points(i);
    for (int point_idx = 0; point_idx < tot_points; point_idx++) {
      float3 point_coord = obj_nurbs_data.calc_nurbs_point_coords(
          i, point_idx, export_params_.scaling_factor);
      fprintf(outfile_, "v %f %f %f\n", point_coord[0], point_coord[1], point_coord[2]);
    }

    const char *nurbs_name = obj_nurbs_data.get_curve_name();
    const int nurbs_degree = obj_nurbs_data.get_nurbs_degree(i);

    fprintf(outfile_, "g %s\ncstype bspline\ndeg %d\n", nurbs_name, nurbs_degree);
    /**
     * curv_num indices into the point vertices above, in relative indices.
     * 0.0 1.0 -1 -2 -3 -4 for a non-cyclic curve with 4 points.
     * 0.0 1.0 -1 -2 -3 -4 -1 -2 -3 for a cyclic curve with 4 points.
     */
    /* Number of vertices in the curve + degree of the curve if it is cyclic. */
    const int curv_num = obj_nurbs_data.get_nurbs_num(i);
    fprintf(outfile_, "curv 0.0 1.0");
    for (int i = 0; i < curv_num; i++) {
      /* + 1 to keep indices one-based, even if they're negative. */
      fprintf(outfile_, " %d", -1 * ((i % tot_points) + 1));
    }
    fprintf(outfile_, "\n");

    /**
     * In parm u line: between 0 and 1, curv_num + 2 equidistant numbers are inserted.
     */
    fprintf(outfile_, "parm u 0.000000 ");
    for (int i = 1; i <= curv_num + 2; i++) {
      fprintf(outfile_, "%f ", 1.0f * i / (curv_num + 2 + 1));
    }
    fprintf(outfile_, "1.000000\n");

    fprintf(outfile_, "end\n");
  }
}

/**
 * When there are multiple objects in a frame, the indices of previous objects' coordinates or
 * normals add up.
 * Make sure to call this after an Object is exported.
 */
void OBJWriter::update_index_offsets(const OBJMesh &obj_mesh_data)
{
  index_offsets_.vertex_offset += obj_mesh_data.tot_vertices();
  index_offsets_.uv_vertex_offset += obj_mesh_data.tot_uv_vertices();
  index_offsets_.normal_offset += per_object_tot_normals_;
  per_object_tot_normals_ = 0;
}

/* -------------------------------------------------------------------- */
/** \name MTL writers.
 * \{ */

/**
 * Converts float3 to space-separated number string with no leading or trailing space.
 * Only to be used in NON performance-critical code.
 */
static std::string float3_to_string(const float3 &numbers)
{
  std::ostringstream r_string;
  r_string << numbers[0] << " " << numbers[1] << " " << numbers[2];
  return r_string.str();
};

/**
 * Open the MTL file in append mode.
 */
MTLWriter::MTLWriter(const char *obj_filepath)
{
  BLI_strncpy(mtl_filepath_, obj_filepath, FILE_MAX);
  BLI_path_extension_replace(mtl_filepath_, FILE_MAX, ".mtl");
  mtl_outfile_ = fopen(mtl_filepath_, "a");
  if (!mtl_outfile_) {
    std::perror(std::string("Error in creating the file at: ").append(mtl_filepath_).c_str());
    return;
  }
  fprintf(stderr, "Material Library: %s\n", mtl_filepath_);
  fprintf(mtl_outfile_, "# Blender %s\n# www.blender.org\n", BKE_blender_version_string());
}

MTLWriter::~MTLWriter()
{
  if (mtl_outfile_ && fclose(mtl_outfile_)) {
    std::cerr << "Error: could not close the MTL file properly, file may be corrupted."
              << std::endl;
  }
}

/**
 * \return if the MTL file is writable.
 */
bool MTLWriter::good() const
{
  return mtl_outfile_ != nullptr;
}

const char *MTLWriter::mtl_file_path() const
{
  BLI_assert(this->good());
  return mtl_filepath_;
}

void MTLWriter::write_bsdf_properties(const blender::io::obj::MTLMaterial &mtl_material)
{
  fprintf(mtl_outfile_,
          "Ni %0.6f\n"
          "d %.6f\n"
          "Ns %0.6f\n"
          "illum %d\n",
          mtl_material.Ni,
          mtl_material.d,
          mtl_material.Ns,
          mtl_material.illum);
  fprintf(mtl_outfile_, "Ka %s\n", float3_to_string(mtl_material.Ka).c_str());
  fprintf(mtl_outfile_, "Kd %s\n", float3_to_string(mtl_material.Kd).c_str());
  fprintf(mtl_outfile_, "Ks %s\n", float3_to_string(mtl_material.Ks).c_str());
  fprintf(mtl_outfile_, "Ke %s\n", float3_to_string(mtl_material.Ke).c_str());
}

/**
 * Write a texture map in the form "map_XX -s 1 1 1 -o 0 0 0 -bm 1 path/to/image" .
 */
void MTLWriter::write_texture_map(const MTLMaterial &mtl_material,
                                  const Map<const std::string, tex_map_XX>::Item &texture_map)
{
  std::string map_bump_strength;
  std::string scale;
  std::string translation;
  /* Optional strings should have their own leading spaces. */
  if (texture_map.value.translation != float3{0.0f, 0.0f, 0.0f}) {
    translation.append(" -s ").append(float3_to_string(texture_map.value.translation));
  }
  if (texture_map.value.scale != float3{1.0f, 1.0f, 1.0f}) {
    scale.append(" -o ").append(float3_to_string(texture_map.value.scale));
  }
  if (texture_map.key == "map_Bump" && mtl_material.map_Bump_strength > 0.0001f) {
    map_bump_strength.append(" -bm ").append(std::to_string(mtl_material.map_Bump_strength));
  }

  /* Always keep only one space between options since filepaths may have leading spaces too. */
  fprintf(mtl_outfile_,
          "%s%s%s%s %s\n",
          texture_map.key.c_str(),
          translation.c_str(),       /* Can be empty. */
          scale.c_str(),             /* Can be empty. */
          map_bump_strength.c_str(), /* Can be empty. */
          texture_map.value.image_path.c_str());
}

void MTLWriter::append_materials(const OBJMesh &mesh_to_export)
{
  BLI_assert(this->good());
  if (!mtl_outfile_) {
    /* Error logging in constructor. */
    return;
  }
  Vector<MTLMaterial> mtl_materials;
  MaterialWrap mat_wrap(mesh_to_export, mtl_materials);
  mat_wrap.fill_materials();

#ifdef DEBUG
  auto all_items_positive = [](const float3 &triplet) {
    return triplet.x >= 0.0f && triplet.y >= 0.0f && triplet.z >= 0.0f;
  };
#endif
  for (const MTLMaterial &mtl_material : mtl_materials) {
    fprintf(mtl_outfile_, "\nnewmtl %s\n", mtl_material.name.c_str());
    BLI_assert(all_items_positive({mtl_material.d, mtl_material.Ns, mtl_material.Ni}) &&
               mtl_material.illum > 0);
    BLI_assert(all_items_positive(mtl_material.Ka) && all_items_positive(mtl_material.Kd) &&
               all_items_positive(mtl_material.Ks) && all_items_positive(mtl_material.Ke));

    write_bsdf_properties(mtl_material);

    /* Write image texture maps. */
    for (const Map<const std::string, tex_map_XX>::Item &texture_map :
         mtl_material.texture_maps.items()) {
      if (texture_map.value.image_path.empty()) {
        continue;
      }
      write_texture_map(mtl_material, texture_map);
    }
  }
}

/** \} */

}  // namespace blender::io::obj
