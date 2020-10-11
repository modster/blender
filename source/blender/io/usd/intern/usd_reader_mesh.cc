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
#include "usd_util.h"

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

#include <iostream>

namespace {

struct MeshSampleData {
  pxr::VtArray<pxr::GfVec3f> points;
  pxr::VtArray<int> vertex_counts;
  pxr::VtArray<int> vertex_indices;
  pxr::VtVec2fArray uv_values;
  pxr::VtArray<int> uv_indices;
  pxr::TfToken uv_interpolation;
  pxr::VtArray<pxr::GfVec3f> normals;
  pxr::TfToken normals_interpolation;

  bool y_up;
  bool reverse_vert_order;
};

}  // anonymous namespace

static const pxr::TfToken st_primvar_name("st", pxr::TfToken::Immortal);

static void sample_uvs(const pxr::UsdGeomMesh &mesh,
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

    st_primvar.GetIndices(&mesh_data.uv_indices, time);
    mesh_data.uv_interpolation = st_primvar.GetInterpolation();
  }
}

static void read_mverts(MVert *mverts, const MeshSampleData &mesh_data)
{
  for (int i = 0; i < mesh_data.points.size(); i++) {
    MVert &mvert = mverts[i];
    pxr::GfVec3f pt = mesh_data.points[i];

    if (mesh_data.y_up) {
      blender::io::usd::copy_zup_from_yup(mvert.co, pt.GetArray());
    }
    else {
      mvert.co[0] = pt[0];
      mvert.co[1] = pt[1];
      mvert.co[2] = pt[2];
    }
  }
}

static void *add_customdata(Mesh *mesh, const char *name, int data_type)
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

static void read_mpolys(Mesh *mesh, const MeshSampleData &mesh_data)
{
  if (!mesh || mesh->totloop == 0) {
    return;
  }

  MPoly *mpolys = mesh->mpoly;
  MLoop *mloops = mesh->mloop;
  MLoopUV *mloopuvs = nullptr;

  const bool do_uvs = (mesh_data.uv_interpolation == pxr::UsdGeomTokens->faceVarying ||
                       mesh_data.uv_interpolation == pxr::UsdGeomTokens->vertex) &&
                      !(mesh_data.uv_indices.empty() && mesh_data.uv_values.empty());

  if (do_uvs) {
    void *cd_ptr = add_customdata(mesh, "uvMap", CD_MLOOPUV);
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

      int loop_index = loop_start;

      if (mesh_data.reverse_vert_order) {
        loop_index += face_size - 1 - f;
      }
      else {
        loop_index += f;
      }

      MLoop &loop = mloops[loop_index];
      loop.v = mesh_data.vertex_indices[loop_index];

      if (mloopuvs) {
        MLoopUV &loopuv = mloopuvs[loop_index];

        int uv_index = mesh_data.uv_interpolation == pxr::UsdGeomTokens->vertex ? loop.v :
                                                                                  loop_index;

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

static void process_no_normals(Mesh *mesh)
{
  /* Absense of normals in the USD mesh is interpreted as 'smooth'. */
  BKE_mesh_calc_normals(mesh);
}

static void process_loop_normals(Mesh *mesh, const MeshSampleData &mesh_data)
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

  for (int i = 0; i < loop_count; ++i) {

    if (mesh_data.y_up) {
      blender::io::usd::copy_zup_from_yup(lnors[i], mesh_data.normals[i].data());
    }
    else {
      lnors[i][0] = mesh_data.normals[i].data()[0];
      lnors[i][1] = mesh_data.normals[i].data()[1];
      lnors[i][2] = mesh_data.normals[i].data()[2];
    }

    if (mesh_data.reverse_vert_order) {
      lnors[i][0] = -lnors[i][0];
      lnors[i][1] = -lnors[i][1];
      lnors[i][2] = -lnors[i][2];
    }
  }

  mesh->flag |= ME_AUTOSMOOTH;
  BKE_mesh_set_custom_normals(mesh, lnors);

  MEM_freeN(lnors);
}

static void process_vertex_normals(Mesh *mesh, const MeshSampleData &mesh_data)
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
      blender::io::usd::copy_zup_from_yup(vnors[i], mesh_data.normals[i].data());
    }
    else {
      vnors[i][0] = mesh_data.normals[i].data()[0];
      vnors[i][1] = mesh_data.normals[i].data()[1];
      vnors[i][2] = mesh_data.normals[i].data()[2];
    }

    if (mesh_data.reverse_vert_order) {
      vnors[i][0] = -vnors[i][0];
      vnors[i][1] = -vnors[i][1];
      vnors[i][2] = -vnors[i][2];
    }
  }

  mesh->flag |= ME_AUTOSMOOTH;
  BKE_mesh_set_custom_normals_from_vertices(mesh, vnors);
  MEM_freeN(vnors);
}

static void process_normals(Mesh *mesh, const MeshSampleData &mesh_data)
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

namespace blender::io::usd {

UsdMeshReader::UsdMeshReader(const pxr::UsdPrim &prim, const USDImporterContext &context)
    : UsdObjectReader(prim, context), mesh_(prim)
{
}

UsdMeshReader::~UsdMeshReader()
{
}

bool UsdMeshReader::valid() const
{
  return static_cast<bool>(mesh_);
}

Mesh *UsdMeshReader::read_mesh(Mesh *existing_mesh,
                               double time,
                               int read_flag,
                               const char **err_str)
{
  if (!this->mesh_) {
    if (err_str) {
      *err_str = "Error reading invalid mesh.";
    }
    return existing_mesh;
  }

  MeshSampleData mesh_data;

  pxr::TfToken orientation;
  mesh_.GetOrientationAttr().Get(&orientation);

  mesh_data.reverse_vert_order = orientation == pxr::UsdGeomTokens->leftHanded;

  mesh_data.y_up = this->context_.stage_up_axis == pxr::UsdGeomTokens->y;

  mesh_.GetPointsAttr().Get(&mesh_data.points, time);
  mesh_.GetFaceVertexCountsAttr().Get(&mesh_data.vertex_counts, time);
  mesh_.GetFaceVertexIndicesAttr().Get(&mesh_data.vertex_indices, time);

  /* For now, always return a new mesh.
   * TODO(makowalski): Add logic to handle the cases where the topology
   * hasn't chaged and we return the existing mesh with updated
   * vert positions. */

  Mesh *new_mesh = nullptr;

  new_mesh = BKE_mesh_new_nomain_from_template(existing_mesh,
                                               mesh_data.points.size(),
                                               0,
                                               0,
                                               mesh_data.vertex_indices.size(),
                                               mesh_data.vertex_counts.size());

  if (read_flag & MOD_MESHSEQ_READ_VERT) {
    read_mverts(new_mesh->mvert, mesh_data);
  }

  if (read_flag & MOD_MESHSEQ_READ_POLY) {
    if ((read_flag & MOD_MESHSEQ_READ_UV) && this->context_.import_params.import_uvs) {
      sample_uvs(mesh_, mesh_data, st_primvar_name, time);
    }

    read_mpolys(new_mesh, mesh_data);

    if (this->context_.import_params.import_normals) {
      mesh_.GetNormalsAttr().Get(&mesh_data.normals, time);
      mesh_data.normals_interpolation = mesh_.GetNormalsInterpolation();

      process_normals(new_mesh, mesh_data);
    }
    else {
      process_no_normals(new_mesh);
    }
  }

  /* TODO(makowalski):  Handle case where topology hasn't changed. */

  return new_mesh;
}

void UsdMeshReader::readObjectData(Main *bmain, double time)
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

  Mesh *mesh = BKE_mesh_add(bmain, prim_name_.c_str());

  std::string obj_name = merged_with_parent_ ? prim_parent_name_ : prim_name_;

  object_ = BKE_object_add_only_object(bmain, OB_MESH, obj_name.c_str());
  object_->data = mesh;

  Mesh *read_mesh = this->read_mesh(mesh, time, MOD_MESHSEQ_READ_ALL, NULL);
  if (read_mesh != mesh) {
    /* XXX fixme after 2.80; mesh->flag isn't copied by BKE_mesh_nomain_to_mesh() */
    /* read_mesh can be freed by BKE_mesh_nomain_to_mesh(), so get the flag before that happens. */
    short autosmooth = (read_mesh->flag & ME_AUTOSMOOTH);
    BKE_mesh_nomain_to_mesh(read_mesh, mesh, object_, &CD_MASK_MESH, true);
    mesh->flag |= autosmooth;
  }

  /* TODO(makowalski):  Read face sets and add modifier. */
}

}  // namespace blender::io::usd
