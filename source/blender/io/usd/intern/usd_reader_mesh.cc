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
 */

#include "usd_reader_mesh.h"
#include "usd_reader_material.h"
#include "usd_reader_prim.h"

#include "MEM_guardedalloc.h"
extern "C" {
#include "DNA_cachefile_types.h"
#include "DNA_camera_types.h"
#include "DNA_constraint_types.h"
#include "DNA_curve_types.h"
#include "DNA_customdata_types.h"
#include "DNA_material_types.h"
#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"
#include "DNA_modifier_types.h"
#include "DNA_object_types.h"
#include "DNA_space_types.h" /* for FILE_MAX */

#include "BKE_camera.h"
#include "BKE_constraint.h"
#include "BKE_curve.h"
#include "BKE_customdata.h"
#include "BKE_main.h"
#include "BKE_material.h"
#include "BKE_mesh.h"
#include "BKE_modifier.h"
#include "BKE_object.h"

#include "BLI_listbase.h"
#include "BLI_math.h"
#include "BLI_math_geom.h"
#include "BLI_string.h"
#include "BLI_utildefines.h"

#include "WM_api.h"
#include "WM_types.h"
}

#include <pxr/base/vt/array.h>
#include <pxr/base/vt/types.h>
#include <pxr/base/vt/value.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/sdf/types.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/subset.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>

#include <iostream>

namespace usdtokens {
// Materials
static const pxr::TfToken st("st", pxr::TfToken::Immortal);
static const pxr::TfToken UVMap("UVMap", pxr::TfToken::Immortal);
static const pxr::TfToken Cd("Cd", pxr::TfToken::Immortal);
static const pxr::TfToken displayColor("displayColor", pxr::TfToken::Immortal);
static const pxr::TfToken normalsPrimvar("normals", pxr::TfToken::Immortal);
}  // namespace usdtokens

namespace utils {
// Very similar to abc mesh utils
static void build_mat_map(const Main *bmain, std::map<std::string, Material *> &mat_map)
{
  Material *material = static_cast<Material *>(bmain->materials.first);

  for (; material; material = static_cast<Material *>(material->id.next)) {
    // We have to do this because the stored material name is coming directly from usd
    mat_map[pxr::TfMakeValidIdentifier(material->id.name + 2).c_str()] = material;
  }
}

static void assign_materials(Main *bmain,
                             Object *ob,
                             const std::map<pxr::SdfPath, int> &mat_index_map,
                             const USDImportParams &params,
                             pxr::UsdStageRefPtr stage)
{
  if (!(stage && bmain && ob)) {
    return;
  }

  bool can_assign = true;
  std::map<pxr::SdfPath, int>::const_iterator it = mat_index_map.begin();

  int matcount = 0;
  for (; it != mat_index_map.end(); ++it, matcount++) {
    if (!BKE_object_material_slot_add(bmain, ob)) {
      can_assign = false;
      break;
    }
  }

  /* TODO(kevin): use global map? */
  std::map<std::string, Material *> mat_map;
  build_mat_map(bmain, mat_map);

  std::map<std::string, Material *>::iterator mat_iter;

  if (can_assign) {
    it = mat_index_map.begin();

    blender::io::usd::USDMaterialReader mat_reader(params, bmain);

    for (; it != mat_index_map.end(); ++it) {
      std::string mat_name = it->first.GetName();
      mat_iter = mat_map.find(mat_name.c_str());

      Material *assigned_mat = nullptr;

      if (mat_iter == mat_map.end()) {

        // Look up the USD material.
        pxr::UsdPrim prim = stage->GetPrimAtPath(it->first);
        pxr::UsdShadeMaterial usd_mat(prim);

        if (usd_mat) {
          assigned_mat = mat_reader.add_material(usd_mat);
          if (assigned_mat) {
            mat_map[mat_name] = assigned_mat;
          }
        }
        else {
          std::cout << "WARNING: Couldn't get USD material " << it->first << std::endl;
        }
      }
      else {
        assigned_mat = mat_iter->second;
      }

      if (assigned_mat) {
        BKE_object_material_assign(bmain, ob, assigned_mat, it->second, BKE_MAT_ASSIGN_OBDATA);
      }
      else {
        std::cout << "WARNING: Couldn't assign material " << mat_name << std::endl;
      }
    }
  }
}

}  // namespace utils

static void *add_customdata_cb(Mesh *mesh, const char *name, int data_type)
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

namespace blender::io::usd {

USDMeshReader::USDMeshReader(pxr::UsdStageRefPtr stage,
                             const pxr::UsdPrim &object,
                             const USDImportParams &import_params,
                             ImportSettings &settings)
    : USDGeomReader(stage, object, import_params, settings),
      m_isLeftHanded(false),
      m_lastNumPositions(-1),
      m_hasUVs(false),
      m_isTimeVarying(false),
      m_isInitialLoad(false)
{
}

void USDMeshReader::createObject(Main *bmain, double motionSampleTime)
{
  Mesh *mesh = BKE_mesh_add(bmain, m_name.c_str());

  m_object = BKE_object_add_only_object(bmain, OB_MESH, m_name.c_str());
  m_object->data = mesh;
}

void USDMeshReader::readObjectData(Main *bmain, double motionSampleTime)
{
  Mesh *mesh = (Mesh *)m_object->data;

  m_isInitialLoad = true;
  Mesh *read_mesh = this->read_mesh(
      mesh, motionSampleTime, m_import_params.global_read_flag, 1.0f, NULL);
  m_isInitialLoad = false;
  if (read_mesh != mesh) {
    /* XXX fixme after 2.80; mesh->flag isn't copied by BKE_mesh_nomain_to_mesh() */
    /* read_mesh can be freed by BKE_mesh_nomain_to_mesh(), so get the flag before that happens. */
    short autosmooth = (read_mesh->flag & ME_AUTOSMOOTH);
    BKE_mesh_nomain_to_mesh(read_mesh, mesh, m_object, &CD_MASK_MESH, true);
    mesh->flag |= autosmooth;
  }

  readFaceSetsSample(bmain, mesh, motionSampleTime);

  if (mesh_prim.GetPointsAttr().ValueMightBeTimeVarying())
    m_isTimeVarying = true;

  if (m_isTimeVarying) {
    addCacheModifier();
  }

  if (m_import_params.import_subdiv) {
    pxr::TfToken subdivScheme;
    mesh_prim.GetSubdivisionSchemeAttr().Get(&subdivScheme, motionSampleTime);

    if (subdivScheme == pxr::UsdGeomTokens->catmullClark) {
      addSubdivModifier();
    }
  }

  USDXformReader::readObjectData(bmain, motionSampleTime);
}

bool USDMeshReader::valid() const
{
  return mesh_prim.GetPrim().IsValid();
}

bool USDMeshReader::topology_changed(Mesh *existing_mesh, double motionSampleTime)
{
  pxr::UsdAttribute faceVertCountsAttr = mesh_prim.GetFaceVertexCountsAttr();
  pxr::UsdAttribute faceVertIndicesAttr = mesh_prim.GetFaceVertexIndicesAttr();
  pxr::UsdAttribute pointsAttr = mesh_prim.GetPointsAttr();
  pxr::UsdAttribute normalsAttr = mesh_prim.GetNormalsAttr();

  faceVertIndicesAttr.Get(&m_face_indices, motionSampleTime);
  faceVertCountsAttr.Get(&m_face_counts, motionSampleTime);
  pointsAttr.Get(&m_positions, motionSampleTime);

  // If 'normals' and 'primvars:normals' are both specified, the latter has precedence.
  pxr::UsdGeomPrimvar primvar = mesh_prim.GetPrimvar(usdtokens::normalsPrimvar);
  if (primvar.HasValue()) {
    primvar.ComputeFlattened(&m_normals, motionSampleTime);
    m_normalInterpolation = primvar.GetInterpolation();
  }
  else {
    mesh_prim.GetNormalsAttr().Get(&m_normals, motionSampleTime);
    m_normalInterpolation = mesh_prim.GetNormalsInterpolation();
  }

  if (m_lastNumPositions != m_positions.size()) {
    m_lastNumPositions = m_positions.size();
    return true;
  }

  return false;
}

void USDMeshReader::read_mpolys(Mesh *mesh, pxr::UsdGeomMesh mesh_prim, double motionSampleTime)
{
  MPoly *mpolys = mesh->mpoly;
  MLoop *mloops = mesh->mloop;

  pxr::UsdAttribute faceVertCountsAttr = mesh_prim.GetFaceVertexCountsAttr();
  pxr::UsdAttribute faceVertIndicesAttr = mesh_prim.GetFaceVertexIndicesAttr();

  pxr::VtIntArray face_counts;
  faceVertCountsAttr.Get(&face_counts, motionSampleTime);
  pxr::VtIntArray face_indices;
  faceVertIndicesAttr.Get(&face_indices, motionSampleTime);

  unsigned int loop_index = 0;
  unsigned int rev_loop_index = 0;

  for (int i = 0; i < face_counts.size(); i++) {
    const int face_size = face_counts[i];

    MPoly &poly = mpolys[i];
    poly.loopstart = loop_index;
    poly.totloop = face_size;
    poly.mat_nr = 0;

    /* Polygons are always assumed to be smooth-shaded. If the Alembic mesh should be flat-shaded,
     * this is encoded in custom loop normals. See T71246. */
    poly.flag |= ME_SMOOTH;

    rev_loop_index = loop_index + (face_size - 1);

    for (int f = 0; f < face_size; f++, loop_index++, rev_loop_index--) {
      MLoop &loop = mloops[loop_index];
      if (m_isLeftHanded)
        loop.v = face_indices[rev_loop_index];
      else
        loop.v = face_indices[loop_index];
    }
  }

  BKE_mesh_calc_edges(mesh, false, false);
}

void USDMeshReader::read_uvs(Mesh *mesh,
                             pxr::UsdGeomMesh mesh_prim,
                             double motionSampleTime,
                             bool load_uvs)
{
  unsigned int loop_index = 0;
  unsigned int rev_loop_index = 0;
  unsigned int uv_index = 0;

  const CustomData *ldata = &mesh->ldata;

  struct UVSample {
    pxr::VtVec2fArray uvs;
    pxr::VtIntArray indices;  // Non-empty for indexed UVs
    pxr::TfToken interpolation;
  };

  std::vector<UVSample> uv_primvars(ldata->totlayer);

  if (m_hasUVs) {
    for (int layer_idx = 0; layer_idx < ldata->totlayer; layer_idx++) {
      const CustomDataLayer *layer = &ldata->layers[layer_idx];
      std::string layer_name = std::string(layer->name);
      if (layer->type != CD_MLOOPUV) {
        continue;
      }

      pxr::TfToken uv_token;

      // If first time seeing uv token, store in map of <layer->uid, TfToken>
      if (uv_token_map.find(layer_name) == uv_token_map.end()) {
        uv_token = pxr::TfToken(layer_name);
        uv_token_map.insert(std::make_pair(layer_name, uv_token));
      }
      else
        uv_token = uv_token_map.at(layer_name);

      // Early out if no token found, this should never happen
      if (uv_token.IsEmpty()) {
        continue;
      }
      // Early out if not first load and uvs arent animated
      if (!load_uvs && primvar_varying_map.find(uv_token) != primvar_varying_map.end() &&
          !primvar_varying_map.at(uv_token)) {
        continue;
      }

      // Early out if mesh doesn't have primvar
      if (!mesh_prim.HasPrimvar(uv_token)) {
        continue;
      }

      if (pxr::UsdGeomPrimvar uv_primvar = mesh_prim.GetPrimvar(uv_token)) {
        uv_primvar.Get<pxr::VtVec2fArray>(&uv_primvars[layer_idx].uvs, motionSampleTime);
        uv_primvar.GetIndices(&uv_primvars[layer_idx].indices, motionSampleTime);
        uv_primvars[layer_idx].interpolation = uv_primvar.GetInterpolation();
      }
    }
  }

  for (int i = 0; i < m_face_counts.size(); i++) {
    const int face_size = m_face_counts[i];

    rev_loop_index = loop_index + (face_size - 1);

    for (int f = 0; f < face_size; f++, loop_index++, rev_loop_index--) {

      for (int layer_idx = 0; layer_idx < ldata->totlayer; layer_idx++) {
        const CustomDataLayer *layer = &ldata->layers[layer_idx];
        if (layer->type != CD_MLOOPUV) {
          continue;
        }

        // Early out if mismatched layer sizes
        if (layer_idx > uv_primvars.size()) {
          continue;
        }

        // Early out if no uvs loaded
        if (uv_primvars[layer_idx].uvs.empty()) {
          continue;
        }

        const UVSample &sample = uv_primvars[layer_idx];

        // For Vertex interpolation, use the vertex index.
        int usd_uv_index = sample.interpolation == pxr::UsdGeomTokens->vertex ?
                               mesh->mloop[loop_index].v :
                               loop_index;

        // Handle indexed UVs.
        usd_uv_index = sample.indices.empty() ? usd_uv_index : sample.indices[usd_uv_index];

        if (usd_uv_index >= sample.uvs.size()) {
          continue;
        }

        MLoopUV *mloopuv = static_cast<MLoopUV *>(layer->data);
        if (m_isLeftHanded)
          uv_index = rev_loop_index;
        else
          uv_index = loop_index;

        mloopuv[uv_index].uv[0] = sample.uvs[usd_uv_index][0];
        mloopuv[uv_index].uv[1] = sample.uvs[usd_uv_index][1];
      }
    }
  }
}

void USDMeshReader::read_attributes(Mesh *mesh,
                                    pxr::UsdGeomMesh mesh_prim,
                                    double motionSampleTime)
{
  CustomData *cd = &mesh->vdata;
  std::vector<pxr::UsdGeomPrimvar> primvars = mesh_prim.GetPrimvars();

  for (pxr::UsdGeomPrimvar p : primvars) {
    int cd_type = 0;
    size_t num = 0;
    size_t type_size = 0;
    void *data;

    if (!p.HasAuthoredValue())
      continue;

    if (!m_isInitialLoad &&
        primvar_varying_map.find(p.GetPrimvarName()) == primvar_varying_map.end())
      continue;

    const char *name = p.GetPrimvarName().GetString().c_str();
    if (p.GetTypeName() == pxr::SdfValueTypeNames->TexCoord2hArray ||
        p.GetTypeName() == pxr::SdfValueTypeNames->TexCoord2fArray ||
        p.GetTypeName() == pxr::SdfValueTypeNames->TexCoord2dArray) {
      continue;
    }
    else if (p.GetTypeName() == pxr::SdfValueTypeNames->Color3fArray ||
             p.GetTypeName() == pxr::SdfValueTypeNames->Double3Array ||
             p.GetTypeName() == pxr::SdfValueTypeNames->Float3Array) {
      cd_type = CD_VALUE_F3;
      pxr::VtVec3fArray idata;
      p.Get<pxr::VtVec3fArray>(&idata, motionSampleTime);
      num = idata.size();
      type_size = sizeof(pxr::GfVec3f);
      data = (void *)idata.cdata();
    }
    else if (p.GetTypeName() == pxr::SdfValueTypeNames->FloatArray ||
             p.GetTypeName() == pxr::SdfValueTypeNames->DoubleArray) {
      cd_type = CD_VALUE_FLOAT;
      pxr::VtFloatArray idata;
      p.Get<pxr::VtFloatArray>(&idata, motionSampleTime);
      num = idata.size();
      type_size = sizeof(float);
      data = (void *)idata.cdata();
    }
    else if (p.GetTypeName() == pxr::SdfValueTypeNames->IntArray) {
      cd_type = CD_VALUE_INT;
      pxr::VtIntArray idata;
      p.Get<pxr::VtIntArray>(&idata, motionSampleTime);
      num = idata.size();
      type_size = sizeof(int);
      data = (void *)idata.cdata();
    }
    else if (p.GetTypeName() == pxr::SdfValueTypeNames->Int3Array) {
      cd_type = CD_VALUE_I3;
      pxr::VtVec3iArray idata;
      p.Get<pxr::VtVec3iArray>(&idata, motionSampleTime);
      num = idata.size();
      type_size = sizeof(pxr::GfVec3i);
      data = (void *)idata.cdata();
    }
    else {
      continue;
    }

    void *cdata = CustomData_get_layer_named(cd, cd_type, name);

    if (!cdata) {
      cdata = CustomData_add_layer_named(cd, cd_type, CD_DEFAULT, NULL, num, name);
    }

    if (cdata) {
      memcpy(cdata, data, num * type_size);
    }
    else {
      std::cerr << "WARNING: Couldn't add custom data layer " << name << std::endl;
    }
  }
}

void USDMeshReader::read_vels(Mesh *mesh,
                              pxr::UsdGeomMesh mesh_prim,
                              float vel_scale,
                              double motionSampleTime)
{
  pxr::VtVec3fArray velocities;
  pxr::UsdAttribute vel_attr = mesh_prim.GetVelocitiesAttr();

  if (!m_isInitialLoad && !vel_attr.ValueMightBeTimeVarying())
    return;

  vel_attr.Get<pxr::VtVec3fArray>(&velocities, motionSampleTime);

  if (velocities.size() <= 0)
    return;

  float(*vdata)[3] = (float(*)[3])CustomData_add_layer(
      &mesh->vdata, CD_VELOCITY, CD_DEFAULT, NULL, mesh->totvert);

  if (vdata) {
    for (int i = 0; i < mesh->totvert; i++) {
      vdata[i][0] = velocities[i][0] * vel_scale;
      vdata[i][1] = velocities[i][1] * vel_scale;
      vdata[i][2] = velocities[i][2] * vel_scale;
    }
  }
}

void USDMeshReader::read_colors(Mesh *mesh,
                                const pxr::UsdGeomMesh &mesh_prim,
                                double motionSampleTime)
{
  if (!(mesh && mesh_prim && mesh->totloop > 0)) {
    return;
  }

  pxr::UsdGeomPrimvar color_primvar = mesh_prim.GetDisplayColorPrimvar();

  if (!color_primvar.HasValue()) {
    return;
  }

  pxr::TfToken interp = color_primvar.GetInterpolation();

  if (interp == pxr::UsdGeomTokens->varying) {
    std::cerr << "WARNING: Unsupported varying interpolation for display colors\n" << std::endl;
    return;
  }

  pxr::VtArray<pxr::GfVec3f> display_colors;

  if (!color_primvar.ComputeFlattened(&display_colors)) {
    std::cerr << "WARNING: Couldn't compute display colors\n" << std::endl;
    return;
  }

  if ((interp == pxr::UsdGeomTokens->faceVarying && display_colors.size() != mesh->totloop) ||
      (interp == pxr::UsdGeomTokens->vertex && display_colors.size() != mesh->totvert) ||
      (interp == pxr::UsdGeomTokens->constant && display_colors.size() != 1) ||
      (interp == pxr::UsdGeomTokens->uniform && display_colors.size() != mesh->totpoly)) {
    std::cerr << "WARNING: display colors count mismatch\n" << std::endl;
    return;
  }

  void *cd_ptr = add_customdata_cb(mesh, "displayColors", CD_MLOOPCOL);

  if (!cd_ptr) {
    std::cerr << "WARNING: Couldn't add displayColors custom data.\n";
    return;
  }

  MLoopCol *colors = static_cast<MLoopCol *>(cd_ptr);

  mesh->mloopcol = colors;

  MPoly *poly = mesh->mpoly;

  for (int i = 0, e = mesh->totpoly; i < e; ++i, ++poly) {
    for (int j = 0; j < poly->totloop; ++j) {
      int loop_index = poly->loopstart + j;

      int usd_index = 0;  // Default for constant varying interpolation.

      if (interp == pxr::UsdGeomTokens->vertex) {
        usd_index = mesh->mloop[loop_index].v;
      }
      else if (interp == pxr::UsdGeomTokens->faceVarying) {
        usd_index = poly->loopstart;
        if (m_isLeftHanded) {
          usd_index += poly->totloop - 1 - j;
        }
        else {
          usd_index += j;
        }
      }
      else if (interp == pxr::UsdGeomTokens->uniform) {
        // Uniform varying uses the poly index.
        usd_index = i;
      }

      if (usd_index >= display_colors.size()) {
        continue;
      }

      colors[loop_index].r = unit_float_to_uchar_clamp(display_colors[usd_index][0]);
      colors[loop_index].g = unit_float_to_uchar_clamp(display_colors[usd_index][1]);
      colors[loop_index].b = unit_float_to_uchar_clamp(display_colors[usd_index][2]);
      colors[loop_index].a = unit_float_to_uchar_clamp(1.0);
    }
  }
}

void USDMeshReader::process_normals_vertex_varying(Mesh *mesh)
{
  if (m_normals.empty()) {
    BKE_mesh_calc_normals(mesh);
    return;
  }

  for (int i = 0; i < m_normals.size(); i++) {
    MVert &mvert = mesh->mvert[i];
    normal_float_to_short_v3(mvert.no, m_normals[i].data());
  }
}

void USDMeshReader::process_normals_face_varying(Mesh *mesh)
{
  if (m_normals.empty()) {
    BKE_mesh_calc_normals(mesh);
    return;
  }

  // Check for normals count mismatches to prevent crashes.
  if (m_normals.size() != mesh->totloop) {
    std::cerr << "WARNING: loop normal count mismatch for mesh " << mesh->id.name << std::endl;
    BKE_mesh_calc_normals(mesh);
    return;
  }

  mesh->flag |= ME_AUTOSMOOTH;

  long int loop_count = m_normals.size();

  float(*lnors)[3] = static_cast<float(*)[3]>(
      MEM_malloc_arrayN(loop_count, sizeof(float[3]), "USD::FaceNormals"));

  MPoly *mpoly = mesh->mpoly;

  for (int i = 0, e = mesh->totpoly; i < e; ++i, ++mpoly) {
    for (int j = 0; j < mpoly->totloop; j++) {
      int blender_index = mpoly->loopstart + j;

      int usd_index = mpoly->loopstart;
      if (m_isLeftHanded) {
        usd_index += mpoly->totloop - 1 - j;
      }
      else {
        usd_index += j;
      }

      lnors[blender_index][0] = m_normals[usd_index][0];
      lnors[blender_index][1] = m_normals[usd_index][1];
      lnors[blender_index][2] = m_normals[usd_index][2];
    }
  }
  BKE_mesh_set_custom_normals(mesh, lnors);

  MEM_freeN(lnors);
}

// Set USD uniform (per-face) normals as Blender loop normals.
void USDMeshReader::process_normals_uniform(Mesh *mesh)
{
  if (m_normals.empty()) {
    BKE_mesh_calc_normals(mesh);
    return;
  }

  // Check for normals count mismatches to prevent crashes.
  if (m_normals.size() != mesh->totpoly) {
    std::cerr << "WARNING: uniform normal count mismatch for mesh " << mesh->id.name << std::endl;
    BKE_mesh_calc_normals(mesh);
    return;
  }

  float(*lnors)[3] = static_cast<float(*)[3]>(
      MEM_malloc_arrayN(mesh->totloop, sizeof(float[3]), "USD::FaceNormals"));

  MPoly *mpoly = mesh->mpoly;

  for (int i = 0, e = mesh->totpoly; i < e; ++i, ++mpoly) {

    for (int j = 0; j < mpoly->totloop; j++) {
      int loop_index = mpoly->loopstart + j;
      lnors[loop_index][0] = m_normals[i][0];
      lnors[loop_index][1] = m_normals[i][1];
      lnors[loop_index][2] = m_normals[i][2];
    }
  }

  mesh->flag |= ME_AUTOSMOOTH;
  BKE_mesh_set_custom_normals(mesh, lnors);

  MEM_freeN(lnors);
}

void USDMeshReader::read_mesh_sample(const std::string &iobject_full_name,
                                     ImportSettings *settings,
                                     Mesh *mesh,
                                     const pxr::UsdGeomMesh &mesh_prim,
                                     double motionSampleTime,
                                     bool new_mesh)
{

  pxr::UsdAttribute normalsAttr = mesh_prim.GetNormalsAttr();
  std::vector<pxr::UsdGeomPrimvar> primvars = mesh_prim.GetPrimvars();
  pxr::UsdAttribute subdivSchemeAttr = mesh_prim.GetSubdivisionSchemeAttr();

  // Note that for new meshes we always want to read verts and polys,
  // regradless of the value of the read_flag, to avoid a crash downstream
  // in code that expect this data to be there.

  if (new_mesh || (settings->read_flag & MOD_MESHSEQ_READ_VERT) != 0) {
    for (int i = 0; i < m_positions.size(); i++) {
      MVert &mvert = mesh->mvert[i];
      mvert.co[0] = m_positions[i][0];
      mvert.co[1] = m_positions[i][1];
      mvert.co[2] = m_positions[i][2];
    }
  }

  if (new_mesh || (settings->read_flag & MOD_MESHSEQ_READ_POLY) != 0) {
    read_mpolys(mesh, mesh_prim, motionSampleTime);
    if (m_normalInterpolation == pxr::UsdGeomTokens->faceVarying) {
      process_normals_face_varying(mesh);
    }
    else if (m_normalInterpolation == pxr::UsdGeomTokens->uniform) {
      process_normals_uniform(mesh);
    }
    else {
      // Default
      BKE_mesh_calc_normals(mesh);
    }
  }

  // Process point normals after reading polys.  This
  // is important in the case where the normals are empty
  // and we invoke BKE_mesh_calc_normals(mesh), which requires
  // edges to be defined.
  if ((settings->read_flag & MOD_MESHSEQ_READ_VERT) != 0 &&
      m_normalInterpolation == pxr::UsdGeomTokens->vertex) {
    process_normals_vertex_varying(mesh);
  }

  if ((settings->read_flag & MOD_MESHSEQ_READ_UV) != 0) {
    read_uvs(mesh, mesh_prim, motionSampleTime, new_mesh);
  }

  if ((settings->read_flag & MOD_MESHSEQ_READ_ATTR) != 0) {
    read_attributes(mesh, mesh_prim, motionSampleTime);
  }

  if ((settings->read_flag & MOD_MESHSEQ_READ_VELS) != 0) {
    read_vels(mesh, mesh_prim, settings->vel_scale, motionSampleTime);
  }

  if ((settings->read_flag & MOD_MESHSEQ_READ_COLOR) != 0) {
    read_colors(mesh, mesh_prim, motionSampleTime);
  }
}

void USDMeshReader::assign_facesets_to_mpoly(double motionSampleTime,
                                             MPoly *mpoly,
                                             int totpoly,
                                             std::map<pxr::SdfPath, int> &r_mat_map)
{
  pxr::UsdShadeMaterialBindingAPI api = pxr::UsdShadeMaterialBindingAPI(m_prim);
  std::vector<pxr::UsdGeomSubset> subsets = api.GetMaterialBindSubsets();

  int current_mat = 0;
  if (subsets.size() > 0) {
    for (pxr::UsdGeomSubset &subset : subsets) {
      pxr::UsdShadeMaterialBindingAPI subsetAPI = pxr::UsdShadeMaterialBindingAPI(
          subset.GetPrim());

      pxr::SdfPath materialPath = subsetAPI.GetDirectBinding().GetMaterialPath();

      if (materialPath.IsEmpty()) {
        continue;
      }

      if (r_mat_map.find(materialPath) == r_mat_map.end()) {
        r_mat_map[materialPath] = 1 + current_mat++;
      }

      const int mat_idx = r_mat_map[materialPath] - 1;

      pxr::UsdAttribute indicesAttribute = subset.GetIndicesAttr();
      pxr::VtIntArray indices;
      indicesAttribute.Get(&indices, motionSampleTime);

      for (int i = 0; i < indices.size(); i++) {
        MPoly &poly = mpoly[indices[i]];
        poly.mat_nr = mat_idx;
      }
    }
  }
  else {
    pxr::SdfPath materialPath = api.GetDirectBinding().GetMaterialPath();
    if (!materialPath.IsEmpty()) {
      r_mat_map[materialPath] = 1;
    }
  }
}

void USDMeshReader::readFaceSetsSample(Main *bmain, Mesh *mesh, const double motionSampleTime)
{
  if (!m_import_params.import_materials) {
    return;
  }

  std::map<pxr::SdfPath, int> mat_map;
  assign_facesets_to_mpoly(motionSampleTime, mesh->mpoly, mesh->totpoly, mat_map);
  utils::assign_materials(bmain, m_object, mat_map, this->m_import_params, this->m_stage);
}

Mesh *USDMeshReader::read_mesh(Mesh *existing_mesh,
                               double motionSampleTime,
                               int read_flag,
                               float vel_scale,
                               const char **err_str)
{
  mesh_prim = pxr::UsdGeomMesh::Get(m_stage, m_prim.GetPath());

  mesh_prim.GetOrientationAttr().Get(&m_orientation);
  if (m_orientation == pxr::UsdGeomTokens->leftHanded)
    m_isLeftHanded = true;

  std::vector<pxr::TfToken> uv_tokens;

  std::vector<pxr::UsdGeomPrimvar> primvars = mesh_prim.GetPrimvars();

  for (pxr::UsdGeomPrimvar p : primvars) {
    if (primvar_varying_map.find(p.GetPrimvarName()) == primvar_varying_map.end()) {
      primvar_varying_map.insert(std::make_pair(p.GetPrimvarName(), p.ValueMightBeTimeVarying()));
      if (p.ValueMightBeTimeVarying())
        m_isTimeVarying = true;
    }

    // Assume all uvs are stored in one of these primvar types...
    if (p.GetTypeName() == pxr::SdfValueTypeNames->TexCoord2hArray ||
        p.GetTypeName() == pxr::SdfValueTypeNames->TexCoord2fArray ||
        p.GetTypeName() == pxr::SdfValueTypeNames->TexCoord2dArray) {
      uv_tokens.push_back(p.GetBaseName());
      m_hasUVs = true;
    }
  }

  Mesh *active_mesh = existing_mesh;
  bool new_mesh = false;

  /* Only read point data when streaming meshes, unless we need to create new ones. */
  ImportSettings settings;
  settings.read_flag |= read_flag;
  settings.vel_scale = vel_scale;

  if (topology_changed(existing_mesh, motionSampleTime)) {
    new_mesh = true;
    active_mesh = BKE_mesh_new_nomain_from_template(
        existing_mesh, m_positions.size(), 0, 0, m_face_indices.size(), m_face_counts.size());

    for (pxr::TfToken token : uv_tokens) {
      void *cd_ptr = add_customdata_cb(active_mesh, token.GetText(), CD_MLOOPUV);
      active_mesh->mloopuv = static_cast<MLoopUV *>(cd_ptr);
    }
  }

  read_mesh_sample(m_prim.GetPath().GetString().c_str(),
                   &settings,
                   active_mesh,
                   mesh_prim,
                   motionSampleTime,
                   new_mesh || m_isInitialLoad);

  if (new_mesh) {
    /* Here we assume that the number of materials doesn't change, i.e. that
     * the material slots that were created when the object was loaded from
     * USD are still valid now. */
    size_t num_polys = active_mesh->totpoly;
    if (num_polys > 0 && m_import_params.import_materials) {
      std::map<pxr::SdfPath, int> mat_map;
      assign_facesets_to_mpoly(motionSampleTime, active_mesh->mpoly, num_polys, mat_map);
    }
  }

  return active_mesh;
}

}  // namespace blender::io::usd
