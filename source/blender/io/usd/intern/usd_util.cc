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

#include "usd_util.h"

#include "usd.h"
#include "usd_hierarchy_iterator.h"
#include "usd_importer_context.h"
#include "usd_reader_object.h"
#include "usd_reader_mesh.h"

#include <pxr/base/plug/registry.h>
#include <pxr/pxr.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/xformable.h>

#include "BKE_main.h"
#include "BKE_material.h"
#include "BKE_mesh.h"
#include "BKE_modifier.h"
#include "BKE_object.h"

#include "BKE_cachefile.h"
#include "BKE_context.h"
#include "BKE_curve.h"
#include "BKE_global.h"
#include "BKE_layer.h"
#include "BKE_lib_id.h"
#include "BKE_object.h"
#include "BKE_scene.h"
#include "BKE_screen.h"

#include "BLI_math_geom.h"

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"

#include <map>

 /* TfToken objects are not cheap to construct, so we do it once. */
namespace usdtokens {
static const pxr::TfToken xform_type("Xform", pxr::TfToken::Immortal);
static const pxr::TfToken mesh_type("Mesh", pxr::TfToken::Immortal);
}  // namespace usdtokens

namespace
{
/* Copy between Z-up and Y-up. */

inline void copy_yup_from_zup(float yup[3], const float zup[3])
{
  const float old_zup1 = zup[1]; /* in case yup == zup */
  yup[0] = zup[0];
  yup[1] = zup[2];
  yup[2] = -old_zup1;
}

inline void copy_zup_from_yup(float zup[3], const float yup[3])
{
  const float old_yup1 = yup[1]; /* in case zup == yup */
  zup[0] = yup[0];
  zup[1] = -yup[2];
  zup[2] = old_yup1;
}

} // end anonymous namespace

namespace blender::io::usd {

void debug_traverse_stage(const pxr::UsdStageRefPtr &usd_stage)
{
  if (!usd_stage)
  {
    return;
  }

  pxr::UsdPrimRange prims = usd_stage->Traverse(pxr::UsdTraverseInstanceProxies(pxr::UsdPrimAllPrimsPredicate));

  for (const pxr::UsdPrim &prim : prims)
  {
    pxr::SdfPath path = prim.GetPath();
    printf("%s\n", path.GetString().c_str());
    printf("  Type: %s\n", prim.GetTypeName().GetString().c_str());
  }
}

void split(const std::string &s, const char delim, std::vector<std::string> &tokens)
{
  tokens.clear();

  std::stringstream ss(s);
  std::string item;

  while (std::getline(ss, item, delim)) {
    if (!item.empty()) {
      tokens.push_back(item);
    }
  }
}

void create_swapped_rotation_matrix(float rot_x_mat[3][3],
                                    float rot_y_mat[3][3],
                                    float rot_z_mat[3][3],
                                    const float euler[3],
                                    UsdAxisSwapMode mode)
{
  const float rx = euler[0];
  float ry;
  float rz;

  /* Apply transformation */
  switch (mode) {
  case USD_ZUP_FROM_YUP:
    ry = -euler[2];
    rz = euler[1];
    break;
  case USD_YUP_FROM_ZUP:
    ry = euler[2];
    rz = -euler[1];
    break;
  default:
    ry = 0.0f;
    rz = 0.0f;
    BLI_assert(false);
    break;
  }

  unit_m3(rot_x_mat);
  unit_m3(rot_y_mat);
  unit_m3(rot_z_mat);

  rot_x_mat[1][1] = cos(rx);
  rot_x_mat[2][1] = -sin(rx);
  rot_x_mat[1][2] = sin(rx);
  rot_x_mat[2][2] = cos(rx);

  rot_y_mat[2][2] = cos(ry);
  rot_y_mat[0][2] = -sin(ry);
  rot_y_mat[2][0] = sin(ry);
  rot_y_mat[0][0] = cos(ry);

  rot_z_mat[0][0] = cos(rz);
  rot_z_mat[1][0] = -sin(rz);
  rot_z_mat[0][1] = sin(rz);
  rot_z_mat[1][1] = cos(rz);
}  // namespace

/* Convert matrix from Z=up to Y=up or vice versa.
 * Use yup_mat = zup_mat for in-place conversion. */
void copy_m44_axis_swap(float dst_mat[4][4], float src_mat[4][4], UsdAxisSwapMode mode)
{
  float dst_rot[3][3], src_rot[3][3], dst_scale_mat[4][4];
  float rot_x_mat[3][3], rot_y_mat[3][3], rot_z_mat[3][3];
  float src_trans[3], dst_scale[3], src_scale[3], euler[3];

  zero_v3(src_trans);
  zero_v3(dst_scale);
  zero_v3(src_scale);
  zero_v3(euler);
  unit_m3(src_rot);
  unit_m3(dst_rot);
  unit_m4(dst_scale_mat);

  /* TODO(Sybren): This code assumes there is no sheer component and no
   * homogeneous scaling component, which is not always true when writing
   * non-hierarchical (e.g. flat) objects (e.g. when parent has non-uniform
   * scale and the child rotates). This is currently not taken into account
   * when axis-swapping. */

   /* Extract translation, rotation, and scale form matrix. */
  mat4_to_loc_rot_size(src_trans, src_rot, src_scale, src_mat);

  /* Get euler angles from rotation matrix. */
  mat3_to_eulO(euler, ROT_MODE_XZY, src_rot);

  /* Create X, Y, Z rotation matrices from euler angles. */
  create_swapped_rotation_matrix(rot_x_mat, rot_y_mat, rot_z_mat, euler, mode);

  /* Concatenate rotation matrices. */
  mul_m3_m3m3(dst_rot, dst_rot, rot_z_mat);
  mul_m3_m3m3(dst_rot, dst_rot, rot_y_mat);
  mul_m3_m3m3(dst_rot, dst_rot, rot_x_mat);

  mat3_to_eulO(euler, ROT_MODE_XZY, dst_rot);

  /* Start construction of dst_mat from rotation matrix */
  unit_m4(dst_mat);
  copy_m4_m3(dst_mat, dst_rot);

  /* Apply translation */
  switch (mode) {
  case USD_ZUP_FROM_YUP:
    copy_zup_from_yup(dst_mat[3], src_trans);
    break;
  case USD_YUP_FROM_ZUP:
    copy_yup_from_zup(dst_mat[3], src_trans);
    break;
  default:
    BLI_assert(false);
  }

  /* Apply scale matrix. Swaps y and z, but does not
   * negate like translation does. */
  dst_scale[0] = src_scale[0];
  dst_scale[1] = src_scale[2];
  dst_scale[2] = src_scale[1];

  size_to_mat4(dst_scale_mat, dst_scale);
  mul_m4_m4m4(dst_mat, dst_mat, dst_scale_mat);
}



void create_readers(const pxr::UsdStageRefPtr &usd_stage, std::vector<UsdObjectReader *> &r_readers, const USDImporterContext &context)
{
  if (!usd_stage)
  {
    return;
  }

  pxr::UsdPrimRange prims = usd_stage->Traverse(pxr::UsdTraverseInstanceProxies(pxr::UsdPrimAllPrimsPredicate));

  for (const pxr::UsdPrim &prim : prims)
  {
    UsdObjectReader *reader = nullptr;

    if (prim.GetTypeName() == usdtokens::mesh_type)
    {
      reader = new UsdMeshReader(prim, context);
    }

    if (reader)
    {
      r_readers.push_back(reader);
      reader->incref();
    }

  }
}

} // namespace blender::io::usd
