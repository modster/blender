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

#include "obj_export_mesh.hh"
#include "obj_export_mtl.hh"
#include "obj_export_nurbs.hh"

#include "obj_export_file_writer.hh"

namespace blender::io::obj {
/**
 * "To turn off smoothing
 * groups, use a value of 0 or off. Polygonal elements use group
 * numbers to put elements in different smoothing groups. For
 * free-form surfaces, smoothing groups are either turned on or off;
 * there is no difference between values greater than 0."
 * http://www.martinreddy.net/gfx/3d/OBJ.spec
 */
const int SMOOTH_GROUP_DISABLED = 0;
const int SMOOTH_GROUP_DEFAULT = 1;

const char *DEFORM_GROUP_DISABLED = "off";
/* There is no deform group default name. Use what the user set in the UI. */

/* "Once a material is assigned, it cannot be turned off; it can only be changed.
 * If a material name is not specified, a white material is used."
 * http://www.martinreddy.net/gfx/3d/OBJ.spec
 * So an empty material name is written. */
const char *MATERIAL_GROUP_DISABLED = "";

/**
 * Write one line of polygon indices as "f v1/vt1/vn1 v2/vt2/vn2 ...".
 */
void OBJWriter::write_vert_uv_normal_indices(Span<int> vert_indices,
                                             Span<int> uv_indices,
                                             Span<int> normal_indices) const
{
  BLI_assert(vert_indices.size() == uv_indices.size() &&
             vert_indices.size() == normal_indices.size());
  file_handler_->write<eOBJSyntaxElement::poly_element_begin>();
  for (int j = 0; j < vert_indices.size(); j++) {
    file_handler_->write<eOBJSyntaxElement::vertex_uv_normal_indices>(
        vert_indices[j] + index_offsets_.vertex_offset + 1,
        uv_indices[j] + index_offsets_.uv_vertex_offset + 1,
        normal_indices[j] + index_offsets_.normal_offset + 1);
  }
  file_handler_->write<eOBJSyntaxElement::poly_element_end>();
}

/**
 * Write one line of polygon indices as "f v1//vn1 v2//vn2 ...".
 */
void OBJWriter::write_vert_normal_indices(Span<int> vert_indices,
                                          Span<int> /*uv_indices*/,
                                          Span<int> normal_indices) const
{
  BLI_assert(vert_indices.size() == normal_indices.size());
  file_handler_->write<eOBJSyntaxElement::poly_element_begin>();
  for (int j = 0; j < vert_indices.size(); j++) {
    file_handler_->write<eOBJSyntaxElement::vertex_normal_indices>(
        vert_indices[j] + index_offsets_.vertex_offset + 1,
        normal_indices[j] + index_offsets_.normal_offset + 1);
  }
  file_handler_->write<eOBJSyntaxElement::poly_element_end>();
}

/**
 * Write one line of polygon indices as "f v1/vt1 v2/vt2 ...".
 */
void OBJWriter::write_vert_uv_indices(Span<int> vert_indices,
                                      Span<int> uv_indices,
                                      Span<int> /*normal_indices*/) const
{
  BLI_assert(vert_indices.size() == uv_indices.size());
  file_handler_->write<eOBJSyntaxElement::poly_element_begin>();
  for (int j = 0; j < vert_indices.size(); j++) {
    file_handler_->write<eOBJSyntaxElement::vertex_uv_indices>(
        vert_indices[j] + index_offsets_.vertex_offset + 1,
        uv_indices[j] + index_offsets_.uv_vertex_offset + 1);
  }
  file_handler_->write<eOBJSyntaxElement::poly_element_end>();
}

/**
 * Write one line of polygon indices as "f v1 v2 ...".
 */
void OBJWriter::write_vert_indices(Span<int> vert_indices,
                                   Span<int> /*uv_indices*/,
                                   Span<int> /*normal_indices*/) const
{
  file_handler_->write<eOBJSyntaxElement::poly_element_begin>();
  for (const int vert_index : vert_indices) {
    file_handler_->write<eOBJSyntaxElement::vertex_indices>(vert_index +
                                                            index_offsets_.vertex_offset + 1);
  }
  file_handler_->write<eOBJSyntaxElement::poly_element_end>();
}

void OBJWriter::write_header() const
{
  using namespace std::string_literals;
  file_handler_->write<eOBJSyntaxElement::string>("# Blender "s + BKE_blender_version_string() +
                                                  "\n");
  file_handler_->write<eOBJSyntaxElement::string>("# www.blender.org\n");
}

/**
 * Write file name of Material Library in .OBJ file.
 */
void OBJWriter::write_mtllib_name(const StringRefNull mtl_filepath) const
{
  /* Split .MTL file path into parent directory and filename. */
  char mtl_file_name[FILE_MAXFILE];
  char mtl_dir_name[FILE_MAXDIR];
  BLI_split_dirfile(mtl_filepath.data(), mtl_dir_name, mtl_file_name, FILE_MAXDIR, FILE_MAXFILE);
  file_handler_->write<eOBJSyntaxElement::mtllib>(mtl_file_name);
}

/**
 * Write an object's group with mesh and/or material name appended conditionally.
 */
void OBJWriter::write_object_group(const OBJMesh &obj_mesh_data) const
{
  /* "o object_name" is not mandatory. A valid .OBJ file may contain neither
   * "o name" nor "g group_name". */
  BLI_assert(export_params_.export_object_groups);
  if (!export_params_.export_object_groups) {
    return;
  }
  const std::string object_name = obj_mesh_data.get_object_name();
  const char *object_mesh_name = obj_mesh_data.get_object_mesh_name();
  const char *object_material_name = obj_mesh_data.get_object_material_name(0);
  if (export_params_.export_materials && export_params_.export_material_groups &&
      object_material_name) {
    file_handler_->write<eOBJSyntaxElement::object_group>(object_name + "_" + object_mesh_name +
                                                          "_" + object_material_name);
    return;
  }
  file_handler_->write<eOBJSyntaxElement::object_group>(object_name + "_" + object_mesh_name);
}

/**
 * Write object's name or group.
 */
void OBJWriter::write_object_name(const OBJMesh &obj_mesh_data) const
{
  const char *object_name = obj_mesh_data.get_object_name();
  if (export_params_.export_object_groups) {
    write_object_group(obj_mesh_data);
    return;
  }
  file_handler_->write<eOBJSyntaxElement::object_name>(object_name);
}

/**
 * Write vertex coordinates for all vertices as "v x y z".
 */
void OBJWriter::write_vertex_coords(const OBJMesh &obj_mesh_data) const
{
  const int tot_vertices = obj_mesh_data.tot_vertices();
  for (int i = 0; i < tot_vertices; i++) {
    float3 vertex = obj_mesh_data.calc_vertex_coords(i, export_params_.scaling_factor);
    file_handler_->write<eOBJSyntaxElement::vertex_coords>(vertex[0], vertex[1], vertex[2]);
  }
}

/**
 * Write UV vertex coordinates for all vertices as "vt u v".
 * \note UV indices are stored here, but written later.
 */
void OBJWriter::write_uv_coords(OBJMesh &r_obj_mesh_data) const
{
  Vector<std::array<float, 2>> uv_coords;
  /* UV indices are calculated and stored in an OBJMesh member here. */
  r_obj_mesh_data.store_uv_coords_and_indices(uv_coords);

  for (const std::array<float, 2> &uv_vertex : uv_coords) {
    file_handler_->write<eOBJSyntaxElement::uv_vertex_coords>(uv_vertex[0], uv_vertex[1]);
  }
}

/**
 * Write loop normals for smooth-shaded polygons, and polygon normals otherwise, as "vn x y z".
 */
void OBJWriter::write_poly_normals(const OBJMesh &obj_mesh_data) const
{
  obj_mesh_data.ensure_mesh_normals();
  Vector<float3> lnormals;
  const int tot_polygons = obj_mesh_data.tot_polygons();
  for (int i = 0; i < tot_polygons; i++) {
    if (obj_mesh_data.is_ith_poly_smooth(i)) {
      obj_mesh_data.calc_loop_normals(i, lnormals);
      for (const float3 &lnormal : lnormals) {
        file_handler_->write<eOBJSyntaxElement::normal>(lnormal[0], lnormal[1], lnormal[2]);
      }
    }
    else {
      float3 poly_normal = obj_mesh_data.calc_poly_normal(i);
      file_handler_->write<eOBJSyntaxElement::normal>(
          poly_normal[0], poly_normal[1], poly_normal[2]);
    }
  }
}

/**
 * Write smooth group if polygon at the given index is shaded smooth else "s 0"
 */
int OBJWriter::write_smooth_group(const OBJMesh &obj_mesh_data,
                                  const int poly_index,
                                  const int last_poly_smooth_group) const
{
  int current_group = SMOOTH_GROUP_DISABLED;
  if (!export_params_.export_smooth_groups && obj_mesh_data.is_ith_poly_smooth(poly_index)) {
    /* Smooth group calculation is disabled, but polygon is smooth-shaded. */
    current_group = SMOOTH_GROUP_DEFAULT;
  }
  else if (obj_mesh_data.is_ith_poly_smooth(poly_index)) {
    /* Smooth group calc is enabled and polygon is smooth–shaded, so find the group. */
    current_group = obj_mesh_data.ith_smooth_group(poly_index);
  }

  if (current_group == last_poly_smooth_group) {
    /* Group has already been written, even if it is "s 0". */
    return current_group;
  }
  file_handler_->write<eOBJSyntaxElement::smooth_group>(current_group);
  return current_group;
}

/**
 * Write material name and material group of a polygon in the .OBJ file.
 * \return #mat_nr of the polygon at the given index.
 * \note It doesn't write to the material library.
 */
int16_t OBJWriter::write_poly_material(const OBJMesh &obj_mesh_data,
                                       const int poly_index,
                                       const int16_t last_poly_mat_nr) const
{
  if (!export_params_.export_materials || obj_mesh_data.tot_materials() <= 0) {
    return last_poly_mat_nr;
  }
  const int16_t current_mat_nr = obj_mesh_data.ith_poly_matnr(poly_index);
  /* Whenever a polygon with a new material is encountered, write its material
   * and/or group, otherwise pass. */
  if (last_poly_mat_nr == current_mat_nr) {
    return current_mat_nr;
  }
  if (current_mat_nr == NOT_FOUND) {
    file_handler_->write<eOBJSyntaxElement::poly_usemtl>(MATERIAL_GROUP_DISABLED);
    return current_mat_nr;
  }
  const char *mat_name = obj_mesh_data.get_object_material_name(current_mat_nr);
  if (export_params_.export_object_groups) {
    write_object_group(obj_mesh_data);
  }
  file_handler_->write<eOBJSyntaxElement::poly_usemtl>(mat_name);
  return current_mat_nr;
}

/**
 * Write the name of the deform group of a polygon.
 */
int16_t OBJWriter::write_vertex_group(const OBJMesh &obj_mesh_data,
                                      const int poly_index,
                                      const int16_t last_poly_vertex_group) const
{
  if (!export_params_.export_vertex_groups) {
    return last_poly_vertex_group;
  }
  const int16_t current_group = obj_mesh_data.get_poly_deform_group_index(poly_index);

  if (current_group == last_poly_vertex_group) {
    /* No vertex group found in this polygon, just like in the last iteration. */
    return current_group;
  }
  if (current_group == NOT_FOUND) {
    file_handler_->write<eOBJSyntaxElement::object_group>(DEFORM_GROUP_DISABLED);
    return current_group;
  }
  file_handler_->write<eOBJSyntaxElement::object_group>(
      obj_mesh_data.get_poly_deform_group_name(current_group));
  return current_group;
}

/**
 * \return Writer function with appropriate polygon-element syntax.
 */
OBJWriter::func_vert_uv_normal_indices OBJWriter::get_poly_element_writer(
    const int total_uv_vertices) const
{
  if (export_params_.export_normals) {
    if (export_params_.export_uv && (total_uv_vertices > 0)) {
      /* Write both normals and UV indices. */
      return &OBJWriter::write_vert_uv_normal_indices;
    }
    /* Write normals indices. */
    return &OBJWriter::write_vert_normal_indices;
  }
  /* Write UV indices. */
  if (export_params_.export_uv && (total_uv_vertices > 0)) {
    return &OBJWriter::write_vert_uv_indices;
  }
  /* Write neither normals nor UV indices. */
  return &OBJWriter::write_vert_indices;
}

/**
 * Write polygon elements with at least vertex indices, and conditionally with UV vertex
 * indices and polygon normal indices. Also write groups: smooth, vertex, material.
 * \note UV indices were stored while writing UV vertices.
 */
void OBJWriter::write_poly_elements(const OBJMesh &obj_mesh_data)
{
  int last_poly_smooth_group = NEGATIVE_INIT;
  int16_t last_poly_vertex_group = NEGATIVE_INIT;
  int16_t last_poly_mat_nr = NEGATIVE_INIT;

  const func_vert_uv_normal_indices poly_element_writer = get_poly_element_writer(
      obj_mesh_data.tot_uv_vertices());

  /* Number of normals may not be equal to number of polygons due to smooth shading. */
  int per_object_tot_normals = 0;
  const int tot_polygons = obj_mesh_data.tot_polygons();
  for (int i = 0; i < tot_polygons; i++) {
    Vector<int> poly_vertex_indices = obj_mesh_data.calc_poly_vertex_indices(i);
    /* For an Object, a normal index depends on how many of its normals have been written before
     * it. This is unknown because of smooth shading. So pass "per object total normals"
     * and update it after each call. */
    int new_normals = 0;
    Vector<int> poly_normal_indices;
    std::tie(new_normals, poly_normal_indices) = obj_mesh_data.calc_poly_normal_indices(
        i, per_object_tot_normals);
    per_object_tot_normals += new_normals;

    last_poly_smooth_group = write_smooth_group(obj_mesh_data, i, last_poly_smooth_group);
    last_poly_vertex_group = write_vertex_group(obj_mesh_data, i, last_poly_vertex_group);
    last_poly_mat_nr = write_poly_material(obj_mesh_data, i, last_poly_mat_nr);
    (this->*poly_element_writer)(
        poly_vertex_indices, obj_mesh_data.uv_indices(i), poly_normal_indices);
  }
  /* Unusual: Other indices are updated in #OBJWriter::update_index_offsets. */
  index_offsets_.normal_offset += per_object_tot_normals;
}

/**
 * Write loose edges of a mesh as "l v1 v2".
 */
void OBJWriter::write_edges_indices(const OBJMesh &obj_mesh_data) const
{
  obj_mesh_data.ensure_mesh_edges();
  const int tot_edges = obj_mesh_data.tot_edges();
  for (int edge_index = 0; edge_index < tot_edges; edge_index++) {
    const std::optional<std::array<int, 2>> vertex_indices =
        obj_mesh_data.calc_loose_edge_vert_indices(edge_index);
    if (!vertex_indices) {
      continue;
    }
    file_handler_->write<eOBJSyntaxElement::edge>(
        (*vertex_indices)[0] + index_offsets_.vertex_offset + 1,
        (*vertex_indices)[1] + index_offsets_.vertex_offset + 1);
  }
}

/**
 * Write a NURBS curve to the .OBJ file in parameter form.
 */
void OBJWriter::write_nurbs_curve(const OBJCurve &obj_nurbs_data) const
{
  const int total_splines = obj_nurbs_data.total_splines();
  for (int spline_idx = 0; spline_idx < total_splines; spline_idx++) {
    const int total_vertices = obj_nurbs_data.total_spline_vertices(spline_idx);
    for (int vertex_idx = 0; vertex_idx < total_vertices; vertex_idx++) {
      const float3 vertex_coords = obj_nurbs_data.vertex_coordinates(
          spline_idx, vertex_idx, export_params_.scaling_factor);
      file_handler_->write<eOBJSyntaxElement::vertex_coords>(
          vertex_coords[0], vertex_coords[1], vertex_coords[2]);
    }

    const char *nurbs_name = obj_nurbs_data.get_curve_name();
    const int nurbs_degree = obj_nurbs_data.get_nurbs_degree(spline_idx);
    file_handler_->write<eOBJSyntaxElement::object_group>(nurbs_name);
    file_handler_->write<eOBJSyntaxElement::cstype>();
    file_handler_->write<eOBJSyntaxElement::nurbs_degree>(nurbs_degree);
    /**
     * The numbers written here are indices into the vertex coordinates written
     * earlier, relative to the line that is going to be written.
     * [0.0 - 1.0] is the curve parameter range.
     * 0.0 1.0 -1 -2 -3 -4 for a non-cyclic curve with 4 vertices.
     * 0.0 1.0 -1 -2 -3 -4 -1 -2 -3 for a cyclic curve with 4 vertices.
     */
    const int total_control_points = obj_nurbs_data.total_spline_control_points(spline_idx);
    file_handler_->write<eOBJSyntaxElement::curve_element_begin>();
    for (int i = 0; i < total_control_points; i++) {
      /* "+1" to keep indices one-based, even if they're negative: i.e., -1 refers to the
       * last vertex coordinate, -2 second last. */
      file_handler_->write<eOBJSyntaxElement::vertex_indices>(-((i % total_vertices) + 1));
    }
    file_handler_->write<eOBJSyntaxElement::curve_element_end>();

    /**
     * In "parm u 0 0.1 .." line:, (total control points + 2) equidistant numbers in the
     * parameter range are inserted.
     */
    file_handler_->write<eOBJSyntaxElement::nurbs_parameter_begin>();
    for (int i = 1; i <= total_control_points + 2; i++) {
      file_handler_->write<eOBJSyntaxElement::nurbs_parameters>(1.0f * i /
                                                                (total_control_points + 2 + 1));
    }
    file_handler_->write<eOBJSyntaxElement::nurbs_parameter_end>();

    file_handler_->write<eOBJSyntaxElement::nurbs_group_end>();
  }
}

/**
 * When there are multiple objects in a frame, the indices of previous objects' coordinates or
 * normals add up.
 */
void OBJWriter::update_index_offsets(const OBJMesh &obj_mesh_data)
{
  index_offsets_.vertex_offset += obj_mesh_data.tot_vertices();
  index_offsets_.uv_vertex_offset += obj_mesh_data.tot_uv_vertices();
  /* Normal index is updated right after writing the normals. */
}

/* -------------------------------------------------------------------- */
/** \name .MTL writers.
 * \{ */

/**
 * Convert #float3 to string of space-separated numbers, with no leading or trailing space.
 * Only to be used in NON-performance-critical code.
 */
static std::string float3_to_string(const float3 &numbers)
{
  std::ostringstream r_string;
  r_string << numbers[0] << " " << numbers[1] << " " << numbers[2];
  return r_string.str();
};

/*
 * Create the .MTL file.
 */
MTLWriter::MTLWriter(const char *obj_filepath) noexcept(false)
{
  mtl_filepath_ = obj_filepath;
  const bool ok = BLI_path_extension_replace(mtl_filepath_.data(), FILE_MAX, ".mtl");
  if (!ok) {
    throw std::system_error(ENAMETOOLONG, std::system_category(), "");
  }
  file_handler_ = std::make_unique<FileHandler<eFileType::MTL>>(mtl_filepath_);
  std::cout << "Material Library created at: " << mtl_filepath_ << std::endl;
}

void MTLWriter::write_header() const
{
  using namespace std::string_literals;
  file_handler_->write<eMTLSyntaxElement::string>("# Blender "s + BKE_blender_version_string() +
                                                  "\n");
  file_handler_->write<eMTLSyntaxElement::string>("# www.blender.org\n");
}

StringRefNull MTLWriter::mtl_file_path() const
{
  return mtl_filepath_;
}

/**
 * Write properties sourced from p-BSDF node or #Object.Material.
 */
void MTLWriter::write_bsdf_properties(const MTLMaterial &mtl_material)
{
  file_handler_->write<eMTLSyntaxElement::Ni>(mtl_material.Ni);
  file_handler_->write<eMTLSyntaxElement::d>(mtl_material.d);
  file_handler_->write<eMTLSyntaxElement::Ns>(mtl_material.Ns);
  file_handler_->write<eMTLSyntaxElement::illum>(mtl_material.illum);
  file_handler_->write<eMTLSyntaxElement::Ka>(
      mtl_material.Ka.x, mtl_material.Ka.y, mtl_material.Ka.z);
  file_handler_->write<eMTLSyntaxElement::Kd>(
      mtl_material.Kd.x, mtl_material.Kd.y, mtl_material.Kd.z);
  file_handler_->write<eMTLSyntaxElement::Ks>(
      mtl_material.Ks.x, mtl_material.Ks.y, mtl_material.Ks.z);
  file_handler_->write<eMTLSyntaxElement::Ke>(
      mtl_material.Ke.x, mtl_material.Ke.y, mtl_material.Ke.z);
}

/**
 * Write a texture map in the form "map_XX -s 1. 1. 1. -o 0. 0. 0. [-bm 1.] path/to/image".
 */
void MTLWriter::write_texture_map(
    const MTLMaterial &mtl_material,
    const Map<const eMTLSyntaxElement, tex_map_XX>::Item &texture_map)
{
  std::string translation;
  std::string scale;
  std::string map_bump_strength;
  /* Optional strings should have their own leading spaces. */
  if (texture_map.value.translation != float3{0.0f, 0.0f, 0.0f}) {
    translation.append(" -s ").append(float3_to_string(texture_map.value.translation));
  }
  if (texture_map.value.scale != float3{1.0f, 1.0f, 1.0f}) {
    scale.append(" -o ").append(float3_to_string(texture_map.value.scale));
  }
  if (texture_map.key == eMTLSyntaxElement::map_Bump && mtl_material.map_Bump_strength > 0.0001f) {
    map_bump_strength.append(" -bm ").append(std::to_string(mtl_material.map_Bump_strength));
  }

#define SYNTAX_DISPATCH(eMTLSyntaxElement) \
  if (texture_map.key == eMTLSyntaxElement) { \
    file_handler_->write<eMTLSyntaxElement>(translation + scale + map_bump_strength, \
                                            texture_map.value.image_path); \
    return; \
  }

  SYNTAX_DISPATCH(eMTLSyntaxElement::map_Kd);
  SYNTAX_DISPATCH(eMTLSyntaxElement::map_Ks);
  SYNTAX_DISPATCH(eMTLSyntaxElement::map_Ns);
  SYNTAX_DISPATCH(eMTLSyntaxElement::map_d);
  SYNTAX_DISPATCH(eMTLSyntaxElement::map_refl);
  SYNTAX_DISPATCH(eMTLSyntaxElement::map_Ke);
  SYNTAX_DISPATCH(eMTLSyntaxElement::map_Bump);

  BLI_assert(!"This map type was not written to the file.");
}

/**
 * Append the materials of the given object to the .MTL file.
 */
void MTLWriter::append_materials(const OBJMesh &mesh_to_export)
{
  MaterialWrap mat_wrap;
  Vector<MTLMaterial> mtl_materials = mat_wrap.fill_materials(mesh_to_export);

#ifndef NDEBUG
  auto all_items_positive = [](const float3 &triplet) {
    return triplet.x >= 0.0f && triplet.y >= 0.0f && triplet.z >= 0.0f;
  };
#endif

  for (const MTLMaterial &mtl_material : mtl_materials) {
    file_handler_->write<eMTLSyntaxElement::newmtl>(mtl_material.name);
    /* At least one material property has not been modified since its initialization. */
    BLI_assert(all_items_positive({mtl_material.d, mtl_material.Ns, mtl_material.Ni}) &&
               mtl_material.illum > 0);
    BLI_assert(all_items_positive(mtl_material.Ka) && all_items_positive(mtl_material.Kd) &&
               all_items_positive(mtl_material.Ks) && all_items_positive(mtl_material.Ke));

    write_bsdf_properties(mtl_material);

    /* Write image texture maps. */
    for (const Map<const eMTLSyntaxElement, tex_map_XX>::Item &texture_map :
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
