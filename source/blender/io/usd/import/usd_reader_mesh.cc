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
    std::cerr << "WARNING: loop normal count mismatch." << std::endl;
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
    std::cerr << "WARNING: vertex normal count mismatch." << std::endl;
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

void build_mtl_map(const Main *bmain, std::map<std::string, Material *> &mat_map)
{
  Material *material = static_cast<Material *>(bmain->materials.first);

  for (; material; material = static_cast<Material *>(material->id.next)) {
    mat_map[material->id.name + 2] = material;
  }
}

}  // anonymous namespace

USDMeshReader::USDMeshReader(const pxr::UsdPrim &prim, const USDImporterContext &context)
    : USDObjectReader(prim, context), mesh_(prim)
{
}

USDMeshReader::~USDMeshReader()
{
}

bool USDMeshReader::valid() const
{
  return static_cast<bool>(mesh_);
}

Mesh *USDMeshReader::read_mesh(Main *bmain, double time)
{
  if (!this->mesh_) {
    std::cerr << "Error reading invalid mesh schema for " << this->prim_path_ << std::endl;
    return nullptr;
  }

  Mesh *mesh = BKE_mesh_add(bmain, prim_name_.c_str());

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
    mesh_.GetNormalsAttr().Get(&mesh_data.normals, time);
    mesh_data.normals_interpolation = mesh_.GetNormalsInterpolation();
    process_normals(mesh, mesh_data);
  }
  else {
    process_no_normals(mesh);
  }

  return mesh;
}

void USDMeshReader::readObjectData(Main *bmain, double time)
{
  if (!this->valid()) {
    return;
  }

  /* Determine mesh visibility.
   * TODO(makowalski): Consider optimizations to avoid this expensive call,
   * for example, by determining visibility during stage traversal. */
  pxr::TfToken vis_tok = this->mesh_.ComputeVisibility();

  if (vis_tok == pxr::UsdGeomTokens->invisible) {
    return;
  }

  std::string obj_name = merged_with_parent_ ? prim_parent_name_ : prim_name_;

  object_ = BKE_object_add_only_object(bmain, OB_MESH, obj_name.c_str());
  Mesh *mesh = this->read_mesh(bmain, time);
  object_->data = mesh;

  if (this->context_.import_params.import_materials) {
    assign_materials(bmain, mesh, time);
  }
}

void USDMeshReader::assign_materials(Main *bmain, Mesh *mesh, double time)
{
  if (!bmain || !mesh || !object_ || !mesh_) {
    return;
  }

  /* Maps USD material names to material instances. */
  std::map<std::string, pxr::UsdShadeMaterial> usd_mtl_map;

  /* Each pair in the following vector represents a subset
   * and the name of the material to which it's bound. */
  std::vector<std::pair<pxr::UsdGeomSubset, std::string>> subset_mtls;

  /* Find the geom subsets that have bound materials.
   * We don't call pxr::UsdShadeMaterialBindingAPI::GetMaterialBindSubsets()
   * because this function returns only those subsets that are in the 'materialBind'
   * family, but, in practice, applications (like Houdini) might export subsets
   * in different families that are bound to materials.
   * TODO(makowalski): Reassess if the above is the best approach. */
  const std::vector<pxr::UsdGeomSubset> face_subsets = pxr::UsdGeomSubset::GetAllGeomSubsets(
      this->mesh_);

  for (const pxr::UsdGeomSubset &sub : face_subsets) {
    pxr::UsdShadeMaterialBindingAPI sub_bind_api(sub);
    PXR_NS::UsdRelationship rel;
    pxr::UsdShadeMaterial sub_bound_mtl = sub_bind_api.ComputeBoundMaterial(
        PXR_NS::UsdShadeTokens->allPurpose, &rel);

    /* Check if we have a bound material that was not inherited from another prim. */
    if (sub_bound_mtl && rel.GetPrim() == sub.GetPrim()) {
      pxr::UsdPrim mtl_prim = sub_bound_mtl.GetPrim();
      if (mtl_prim) {
        std::string mtl_name = sub_bound_mtl.GetPrim().GetName().GetString();
        subset_mtls.push_back(std::make_pair(sub, mtl_name));
        usd_mtl_map.insert(std::make_pair(mtl_name, sub_bound_mtl));
      }
    }
  }

  if (subset_mtls.empty()) {
    /* No material subsets.  See if there is a material bound to the mesh. */

    pxr::UsdShadeMaterialBindingAPI binding_api(this->mesh_.GetPrim());
    pxr::UsdShadeMaterial bound_mtl = binding_api.ComputeBoundMaterial();

    if (!bound_mtl || !bound_mtl.GetPrim()) {
      return;
    }

    /* We have a material bound to the mesh prim. */

    /* Add a material slot to the object .*/

    if (!BKE_object_material_slot_add(bmain, object_)) {
      std::cerr << "WARNING:  Couldn't add material slot for mesh prim " << this->prim_path_
                << std::endl;
      return;
    }

    /* Check if a material with the same name already exists. */

    std::string mtl_name = bound_mtl.GetPrim().GetName().GetString();

    Material *mtl = static_cast<Material *>(bmain->materials.first);
    Material *blen_mtl = nullptr;

    for (; mtl; mtl = static_cast<Material *>(mtl->id.next)) {
      if (strcmp(mtl_name.c_str(), mtl->id.name + 2) == 0) {
        /* Found an existing material with the same name. */
        blen_mtl = mtl;
        break;
      }
    }

    if (!blen_mtl) {
      /* No existing material, so add it now. */
      blen_mtl = BKE_material_add(bmain, mtl_name.c_str());
    }

    if (!blen_mtl) {
      std::cerr << "WARNING:  Couldn't add material " << mtl_name << " for mesh prim "
                << this->prim_path_ << std::endl;
    }

    /* Set the material IDs on the polys. */
    for (int p = 0; p < mesh->totpoly; ++p) {
      mesh->mpoly[p].mat_nr = 0;
    }

    BKE_object_material_assign(bmain, object_, blen_mtl, 1, BKE_MAT_ASSIGN_OBDATA);
  }
  else {

    /* Maps USD material names to material slot index. */
    std::map<std::string, int> mtl_index_map;

    /* Add material slots. */
    std::map<std::string, pxr::UsdShadeMaterial>::const_iterator usd_mtl_iter =
        usd_mtl_map.begin();

    for (; usd_mtl_iter != usd_mtl_map.end(); ++usd_mtl_iter) {
      if (!BKE_object_material_slot_add(bmain, object_)) {
        std::cerr << "WARNING:  Couldn't add material slot for mesh prim " << this->prim_path_
                  << std::endl;
        return;
      }
    }

    /* Create the Blender materials. */

    /* Query the current materials. */
    std::map<std::string, Material *> blen_mtl_map;
    build_mtl_map(bmain, blen_mtl_map);

    /* Iterate over the USD materials and add corresponding
     * Blender materials of the same name, if they don't
     * already exist. */
    usd_mtl_iter = usd_mtl_map.begin();
    int idx = 0;

    for (; usd_mtl_iter != usd_mtl_map.end(); ++usd_mtl_iter, ++idx) {
      Material *blen_mtl = nullptr;

      std::string mtl_name = usd_mtl_iter->first.c_str();

      std::map<std::string, Material *>::const_iterator blen_mtl_iter = blen_mtl_map.find(
          mtl_name);

      if (blen_mtl_iter != blen_mtl_map.end()) {
        blen_mtl = blen_mtl_iter->second;
      }
      else {
        blen_mtl = BKE_material_add(bmain, mtl_name.c_str());

        if (blen_mtl) {
          blen_mtl_map.insert(std::make_pair(mtl_name, blen_mtl));
        }
      }

      if (!blen_mtl) {
        std::cerr << "WARNING:  Couldn't add material " << mtl_name << " for mesh prim "
                  << this->prim_path_ << std::endl;
        return;
      }

      BKE_object_material_assign(bmain, object_, blen_mtl, idx + 1, BKE_MAT_ASSIGN_OBDATA);

      /* Record this material's index. */
      mtl_index_map.insert(std::make_pair(usd_mtl_iter->first, idx));
    }

    /* Assign the material indices. */

    for (const std::pair<pxr::UsdGeomSubset, std::string> &sub_mtl : subset_mtls) {

      /* Find the index of the current material. */
      std::map<std::string, int>::const_iterator mtl_index_iter = mtl_index_map.find(
          sub_mtl.second);

      if (mtl_index_iter == mtl_index_map.end()) {
        std::cerr << "WARNING:  Couldn't find material index." << std::endl;
        return;
      }

      int mtl_idx = mtl_index_iter->second;

      /* Query the subset membership. */
      pxr::VtIntArray indices;
      sub_mtl.first.GetIndicesAttr().Get(&indices, time);

      /* Assign the poly material indices. */
      for (int face_idx : indices) {
        if (mtl_idx > mesh->totpoly) {
          std::cerr << "WARNING:  Out of bounds material index." << std::endl;
          return;
        }
        mesh->mpoly[face_idx].mat_nr = mtl_idx;
      }
    }
  }
}

}  // namespace blender::io::usd
