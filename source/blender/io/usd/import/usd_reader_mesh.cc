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
 * The Original Code is Copyright (C) 2029 Blender Foundation.
 * All rights reserved.
 */

#include "usd_reader_mesh.h"
#include "usd_import_util.h"
#include "usd_material_importer.h"

#include "MEM_guardedalloc.h"

#include "DNA_material_types.h"
#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"
#include "DNA_object_types.h"

#include "BLI_compiler_compat.h"
#include "BLI_math_geom.h"

#include "BKE_main.h"
#include "BKE_material.h"
#include "BKE_mesh.h"
#include "BKE_modifier.h"
#include "BKE_object.h"

#include <pxr/usd/usdShade/materialBindingAPI.h>

#include <iostream>

namespace blender::io::usd {

/* Anonymous namespace for helper functions and definitions. */
namespace {

struct MeshSampleData {
  pxr::VtArray<pxr::GfVec3f> points;
  pxr::VtArray<int> vertex_counts;
  pxr::VtArray<int> vertex_indices;
  pxr::TfToken uv_primvar_name;
  pxr::VtVec2fArray uv_values;
  pxr::VtArray<int> uv_indices;
  pxr::TfToken uv_interpolation;
  pxr::VtArray<pxr::GfVec3f> normals;
  pxr::TfToken normals_interpolation;

  bool y_up;
  bool reverse_vert_order;
};

const pxr::TfToken st_primvar_name("st", pxr::TfToken::Immortal);
const pxr::TfToken normals_primvar_name("normals", pxr::TfToken::Immortal);

void sample_uvs(const pxr::UsdGeomMesh &mesh,
                MeshSampleData &mesh_data,
                pxr::TfToken primvar_name,
                double time)
{
  if (!mesh) {
    return;
  }

  pxr::UsdGeomPrimvar st_primvar = mesh.GetPrimvar(primvar_name);

  if (!st_primvar) {
    return;
  }

  if (st_primvar.GetTypeName() == pxr::SdfValueTypeNames->TexCoord2fArray ||
      st_primvar.GetTypeName() == pxr::SdfValueTypeNames->Float2Array) {
    if (!st_primvar.Get(&mesh_data.uv_values, time)) {
      std::cerr << "WARNING: Couldn't get uvs from primvar " << primvar_name << " for prim "
                << mesh.GetPath() << std::endl;
    }

    mesh_data.uv_primvar_name = primvar_name;
    st_primvar.GetIndices(&mesh_data.uv_indices, time);
    mesh_data.uv_interpolation = st_primvar.GetInterpolation();
  }
}

void read_mverts(MVert *mverts, const MeshSampleData &mesh_data)
{
  for (int i = 0; i < mesh_data.points.size(); i++) {
    MVert &mvert = mverts[i];
    pxr::GfVec3f pt = mesh_data.points[i];

    if (mesh_data.y_up) {
      copy_zup_from_yup(mvert.co, pt.GetArray());
    }
    else {
      mvert.co[0] = pt[0];
      mvert.co[1] = pt[1];
      mvert.co[2] = pt[2];
    }
  }
}

void *add_customdata(Mesh *mesh, const char *name, int data_type)
{
  CustomDataType cd_data_type = static_cast<CustomDataType>(data_type);
  void *cd_ptr;
  CustomData *loopdata;
  int numloops;

  /* unsupported custom data type -- don't do anything. */
  if (!ELEM(cd_data_type, CD_MLOOPUV, CD_MLOOPCOL)) {
    return NULL;
  }

  loopdata = &mesh->ldata;
  cd_ptr = CustomData_get_layer_named(loopdata, cd_data_type, name);
  if (cd_ptr != NULL) {
    /* layer already exists, so just return it. */
    return cd_ptr;
  }

  /* Create a new layer. */
  numloops = mesh->totloop;
  cd_ptr = CustomData_add_layer_named(loopdata, cd_data_type, CD_DEFAULT, NULL, numloops, name);
  return cd_ptr;
}

void read_mpolys(Mesh *mesh, const MeshSampleData &mesh_data)
{
  if (!mesh || mesh->totloop == 0) {
    return;
  }

  MPoly *mpolys = mesh->mpoly;
  MLoop *mloops = mesh->mloop;
  MLoopUV *mloopuvs = nullptr;

  const bool do_uvs = (mesh_data.uv_interpolation == pxr::UsdGeomTokens->faceVarying ||
                       mesh_data.uv_interpolation == pxr::UsdGeomTokens->vertex) &&
                      !(mesh_data.uv_indices.empty() && mesh_data.uv_values.empty()) &&
                      !mesh_data.uv_primvar_name.IsEmpty();

  if (do_uvs) {
    void *cd_ptr = add_customdata(mesh, mesh_data.uv_primvar_name.GetString().c_str(), CD_MLOOPUV);
    mloopuvs = static_cast<MLoopUV *>(cd_ptr);
  }

  int loop_start = 0;

  for (int i = 0; i < mesh_data.vertex_counts.size(); i++) {
    const int face_size = mesh_data.vertex_counts[i];

    MPoly &poly = mpolys[i];
    poly.loopstart = loop_start;
    poly.totloop = face_size;

    poly.flag |= ME_SMOOTH;

    for (int f = 0; f < face_size; ++f) {

      int loop_index = loop_start + f;

      /* Index into the USD data, corresponding to the loop index,
       * but taking reversed winding order into account. */
      int usd_index = loop_start;

      if (mesh_data.reverse_vert_order) {
        usd_index += face_size - 1 - f;
      }
      else {
        usd_index += f;
      }

      MLoop &loop = mloops[loop_index];
      loop.v = mesh_data.vertex_indices[usd_index];

      if (mloopuvs) {
        MLoopUV &loopuv = mloopuvs[loop_index];

        int uv_index = mesh_data.uv_interpolation == pxr::UsdGeomTokens->vertex ? loop.v :
                                                                                  usd_index;

        uv_index = mesh_data.uv_indices.empty() ? uv_index : mesh_data.uv_indices[uv_index];

        /* Range check. */
        if (uv_index >= mesh_data.uv_values.size()) {
          std::cerr << "WARNING:  Out of bounds uv index " << uv_index << std::endl;
          continue;
        }

        loopuv.uv[0] = mesh_data.uv_values[uv_index][0];
        loopuv.uv[1] = mesh_data.uv_values[uv_index][1];
      }
    }

    loop_start += face_size;
  }

  BKE_mesh_calc_edges(mesh, false, false);

  /* TODO(makowalski):  Possibly check for invalid geometry. */
}

void process_no_normals(Mesh *mesh)
{
  /* Absense of normals in the USD mesh is interpreted as 'smooth'. */
  BKE_mesh_calc_normals(mesh);
}

void process_loop_normals(Mesh *mesh, const MeshSampleData &mesh_data)
{
  if (!mesh) {
    return;
  }

  size_t loop_count = mesh_data.normals.size();

  if (loop_count == 0) {
    process_no_normals(mesh);
    return;
  }

  if (loop_count != mesh->totloop) {
    std::cerr << "WARNING: loop normal count mismatch for mesh " << mesh->id.name << std::endl;
    process_no_normals(mesh);
    return;
  }

  float(*lnors)[3] = static_cast<float(*)[3]>(
      MEM_malloc_arrayN(loop_count, sizeof(float[3]), "USD::FaceNormals"));

  MPoly *mpoly = mesh->mpoly;

  for (int p = 0; p < mesh->totpoly; ++p, ++mpoly) {

    for (int l = 0; l < mpoly->totloop; ++l) {
      int blender_index = mpoly->loopstart + l;
      int usd_index = mpoly->loopstart;

      if (mesh_data.reverse_vert_order) {
        usd_index += mpoly->totloop - 1 - l;
      }
      else {
        usd_index += l;
      }

      if (mesh_data.y_up) {
        copy_zup_from_yup(lnors[blender_index], mesh_data.normals[usd_index].data());
      }
      else {
        lnors[blender_index][0] = mesh_data.normals[usd_index].data()[0];
        lnors[blender_index][1] = mesh_data.normals[usd_index].data()[1];
        lnors[blender_index][2] = mesh_data.normals[usd_index].data()[2];
      }
    }
  }

  mesh->flag |= ME_AUTOSMOOTH;
  BKE_mesh_set_custom_normals(mesh, lnors);

  MEM_freeN(lnors);
}

void process_vertex_normals(Mesh *mesh, const MeshSampleData &mesh_data)
{
  if (!mesh) {
    return;
  }

  size_t normals_count = mesh_data.normals.size();
  if (normals_count == 0) {
    process_no_normals(mesh);
    return;
  }

  if (normals_count != mesh->totvert) {
    std::cerr << "WARNING: vertex normal count mismatch for mesh " << mesh->id.name << std::endl;
    process_no_normals(mesh);
    return;
  }

  float(*vnors)[3] = static_cast<float(*)[3]>(
      MEM_malloc_arrayN(normals_count, sizeof(float[3]), "USD::VertexNormals"));

  for (int i = 0; i < normals_count; ++i) {

    if (mesh_data.y_up) {
      copy_zup_from_yup(vnors[i], mesh_data.normals[i].data());
    }
    else {
      vnors[i][0] = mesh_data.normals[i].data()[0];
      vnors[i][1] = mesh_data.normals[i].data()[1];
      vnors[i][2] = mesh_data.normals[i].data()[2];
    }
  }

  mesh->flag |= ME_AUTOSMOOTH;
  BKE_mesh_set_custom_normals_from_vertices(mesh, vnors);
  MEM_freeN(vnors);
}

void process_normals(Mesh *mesh, const MeshSampleData &mesh_data)
{
  if (!mesh || mesh_data.normals.empty()) {
    process_no_normals(mesh);
    return;
  }

  if (mesh_data.normals_interpolation == pxr::UsdGeomTokens->faceVarying) {
    process_loop_normals(mesh, mesh_data); /* 'vertex normals' in Houdini. */
  }
  else if (mesh_data.normals_interpolation == pxr::UsdGeomTokens->vertex) {
    process_vertex_normals(mesh, mesh_data); /* 'point normals' in Houdini. */
  }
  else {
    process_no_normals(mesh);
  }
}

void build_mat_map(const Main *bmain, std::map<std::string, Material *> &mat_map)
{
  Material *material = static_cast<Material *>(bmain->materials.first);

  for (; material; material = static_cast<Material *>(material->id.next)) {
    mat_map[material->id.name + 2] = material;
  }
}

}  // anonymous namespace

USDMeshReader::USDMeshReader(const pxr::UsdPrim &prim, const USDImporterContext &context)
    : USDMeshReaderBase(prim, context), mesh_(prim)
{
}

USDMeshReader::~USDMeshReader()
{
}

bool USDMeshReader::valid() const
{
  return static_cast<bool>(mesh_);
}

Mesh *USDMeshReader::create_mesh(Main *bmain, double time)
{
  if (!this->mesh_) {
    std::cerr << "Error reading invalid mesh schema for " << this->prim_path_ << std::endl;
    return nullptr;
  }

  std::string mesh_name = this->get_data_name();

  if (mesh_name.empty()) {
    /* Sanity check. */
    std::cerr << "Warning: couldn't determine mesh name for " << this->prim_path() << std::endl;
  }

  Mesh *mesh = BKE_mesh_add(bmain, mesh_name.c_str());

  MeshSampleData mesh_data;

  pxr::TfToken orientation;
  mesh_.GetOrientationAttr().Get(&orientation);

  mesh_data.reverse_vert_order = orientation == pxr::UsdGeomTokens->leftHanded;

  mesh_data.y_up = this->context_.stage_up_axis == pxr::UsdGeomTokens->y;

  mesh_.GetPointsAttr().Get(&mesh_data.points, time);

  if (mesh_data.points.empty()) {
    return mesh;
  }

  mesh_.GetFaceVertexCountsAttr().Get(&mesh_data.vertex_counts, time);
  mesh_.GetFaceVertexIndicesAttr().Get(&mesh_data.vertex_indices, time);

  mesh->totvert = mesh_data.points.size();
  mesh->mvert = (MVert *)CustomData_add_layer(
      &mesh->vdata, CD_MVERT, CD_CALLOC, NULL, mesh_data.points.size());

  mesh->totpoly = mesh_data.vertex_counts.size();
  mesh->totloop = mesh_data.vertex_indices.size();
  mesh->mpoly = (MPoly *)CustomData_add_layer(
      &mesh->pdata, CD_MPOLY, CD_CALLOC, NULL, mesh->totpoly);
  mesh->mloop = (MLoop *)CustomData_add_layer(
      &mesh->ldata, CD_MLOOP, CD_CALLOC, NULL, mesh->totloop);

  read_mverts(mesh->mvert, mesh_data);

  if (this->context_.import_params.import_uvs) {
    sample_uvs(mesh_, mesh_data, st_primvar_name, time);
  }

  read_mpolys(mesh, mesh_data);

  if (this->context_.import_params.import_normals) {

    /* If 'normals' and 'primvars:normals' are both specified, the latter has precedence. */
    pxr::UsdGeomPrimvar primvar = mesh_.GetPrimvar(normals_primvar_name);
    if (primvar.HasValue()) {
      primvar.ComputeFlattened(&mesh_data.normals, time);
      mesh_data.normals_interpolation = primvar.GetInterpolation();
    }
    else {
      mesh_.GetNormalsAttr().Get(&mesh_data.normals, time);
      mesh_data.normals_interpolation = mesh_.GetNormalsInterpolation();
    }

    process_normals(mesh, mesh_data);
  }
  else {
    process_no_normals(mesh);
  }

  return mesh;
}

void USDMeshReader::assign_materials(Main *bmain,
                                     Mesh *mesh,
                                     double time,
                                     bool set_object_materials)
{
  if (!mesh && !(set_object_materials && bmain && this->object_ && this->prim())) {
    return;
  }

  std::map<pxr::SdfPath, int> mat_map;

  /* Find the geom subsets that have bound materials.
   * We don't call pxr::UsdShadeMaterialBindingAPI::GetMaterialBindSubsets()
   * because this function returns only those subsets that are in the 'materialBind'
   * family, but, in practice, applications (like Houdini) might export subsets
   * in different families that are bound to materials.
   * TODO(makowalski): Reassess if the above is the best approach. */
  const std::vector<pxr::UsdGeomSubset> face_subsets = pxr::UsdGeomSubset::GetAllGeomSubsets(
      this->mesh_);

  if (!face_subsets.empty()) {

    int current_mat = 0;

    for (const pxr::UsdGeomSubset &sub : face_subsets) {
      pxr::UsdShadeMaterialBindingAPI sub_bind_api(sub);

      /* TODO(makowalski): Verify that the following will work for instance proxies. */

      pxr::SdfPath mat_path = sub_bind_api.GetDirectBinding().GetMaterialPath();

      if (mat_path.IsEmpty()) {
        continue;
      }

      if (mat_map.find(mat_path) == mat_map.end()) {
        mat_map[mat_path] = 1 + current_mat++;
      }

      if (mesh) {
        const int mat_idx = mat_map[mat_path] - 1;

        /* Query the subset membership. */
        pxr::UsdAttribute indicesAttribute = sub.GetIndicesAttr();
        pxr::VtIntArray indices;
        indicesAttribute.Get(&indices, time);

        /* Assign the poly material indices. */
        for (int face_idx : indices) {
          if (face_idx > mesh->totpoly) {
            std::cerr << "WARNING:  Out of bounds material subset index." << std::endl;
            continue;
          }
          mesh->mpoly[face_idx].mat_nr = mat_idx;
        }
      }
    }
  }

  if (!(set_object_materials && bmain && this->object_)) {
    return;
  }

  if (mat_map.empty()) {

    pxr::UsdShadeMaterialBindingAPI binding_api(this->prim());

    // Note that calling binding_api.GetDirectBinding() doesn't appear
    // to work for instance proxies.
    pxr::UsdShadeMaterial bound_mtl = binding_api.ComputeBoundMaterial();

    if (bound_mtl) {
      mat_map.insert(std::make_pair(bound_mtl.GetPath(), 1));
    }
  }

  if (mat_map.empty()) {
    return;
  }

  /* Add material slots. */
  std::map<pxr::SdfPath, int>::const_iterator mat_iter = mat_map.begin();

  for (; mat_iter != mat_map.end(); ++mat_iter) {
    if (!BKE_object_material_slot_add(bmain, object_)) {
      std::cerr << "WARNING:  Couldn't add material slot for mesh prim " << this->prim_path_
                << std::endl;
      return;
    }
  }

  USDMaterialImporter mat_importer(this->context_, bmain);

  /* TODO(makowalski): Move more of the material creation logic into USDMaterialImporter. */

  /* Create the Blender materials. */

  /* Query the current Blender materials. */
  std::map<std::string, Material *> blen_mat_map;
  build_mat_map(bmain, blen_mat_map);

  /* Iterate over the USD materials and add corresponding
   * Blender materials of the same name, if they don't
   * already exist. */
  mat_iter = mat_map.begin();

  for (; mat_iter != mat_map.end(); ++mat_iter) {
    Material *blen_mtl = nullptr;

    std::string mat_name = mat_iter->first.GetName();

    std::map<std::string, Material *>::const_iterator blen_mat_iter = blen_mat_map.find(mat_name);

    if (blen_mat_iter != blen_mat_map.end()) {
      blen_mtl = blen_mat_iter->second;
    }
    else {

      pxr::UsdPrim mat_prim = this->prim().GetStage()->GetPrimAtPath(mat_iter->first);

      pxr::UsdShadeMaterial usd_mat(mat_prim);
      blen_mtl = mat_importer.add_material(usd_mat);

      if (blen_mtl) {
        blen_mat_map.insert(std::make_pair(mat_name, blen_mtl));
      }
    }

    if (!blen_mtl) {
      std::cerr << "WARNING:  Couldn't add material " << mat_name << " for mesh prim "
                << this->prim_path_ << std::endl;
      return;
    }

    BKE_object_material_assign(bmain, object_, blen_mtl, mat_iter->second, BKE_MAT_ASSIGN_OBDATA);
  }
}

}  // namespace blender::io::usd
