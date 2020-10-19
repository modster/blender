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

#include "usd_import_util.h"

#include "usd.h"
#include "usd_importer_context.h"
#include "usd_reader_mesh.h"
#include "usd_reader_object.h"
#include "usd_reader_transform.h"

#include <pxr/base/plug/registry.h>
#include <pxr/pxr.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/scope.h>
#include <pxr/usd/usdGeom/tokens.h>
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

#include <iostream>
#include <map>

/* TfToken objects are not cheap to construct, so we do it once. */
namespace usdtokens {
static const pxr::TfToken xform_type("Xform", pxr::TfToken::Immortal);
static const pxr::TfToken mesh_type("Mesh", pxr::TfToken::Immortal);
static const pxr::TfToken scope_type("Scope", pxr::TfToken::Immortal);
}  // namespace usdtokens

namespace {
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

}  // end anonymous namespace

namespace blender::io::usd {

static UsdObjectReader *get_reader(const pxr::UsdPrim &prim, const USDImporterContext &context)
{
  UsdObjectReader *result = nullptr;

  if (prim.IsA<pxr::UsdGeomMesh>()) {
    result = new UsdMeshReader(prim, context);
  }
  else if (prim.IsA<pxr::UsdGeomXform>()) {
    result = new UsdTransformReader(prim, context);
  }

  return result;
}

void debug_traverse_stage(const pxr::UsdStageRefPtr &usd_stage)
{
  if (!usd_stage) {
    return;
  }

  pxr::UsdPrimRange prims = usd_stage->Traverse(
      pxr::UsdTraverseInstanceProxies(pxr::UsdPrimAllPrimsPredicate));

  for (const pxr::UsdPrim &prim : prims) {
    pxr::SdfPath path = prim.GetPath();
    printf("%s\n", path.GetString().c_str());
    printf("  Type: %s\n", prim.GetTypeName().GetString().c_str());
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

void create_readers(const pxr::UsdStageRefPtr &usd_stage,
                    const USDImporterContext &context,
                    std::vector<UsdObjectReader *> &r_readers)
{
  if (!usd_stage) {
    return;
  }

  /* Map a USD prim path to the corresponding reader,
   * for keeping track of which prims have been processed
   * and for setting parenting relationships when we are
   * done with the traversal. */
  std::map<std::string, UsdObjectReader *> readers_map;

  pxr::UsdPrimRange prims = usd_stage->Traverse(
      pxr::UsdTraverseInstanceProxies(pxr::UsdPrimAllPrimsPredicate));

  for (const pxr::UsdPrim &prim : prims) {

    std::string prim_path = prim.GetPath().GetString();

    std::map<std::string, UsdObjectReader *>::const_iterator prim_entry = readers_map.find(
        prim_path);

    if (prim_entry != readers_map.end()) {
      /* We already processed the reader for this prim, probably when merging it with its parent.
       */
      continue;
    }

    UsdObjectReader *reader = nullptr;
    bool merge_reader = false;

    if (prim.GetTypeName() == usdtokens::xform_type) {

      /* Check if the Xform and prim should be merged. */

      pxr::UsdPrimSiblingRange children = prim.GetFilteredChildren(
          pxr::UsdTraverseInstanceProxies(pxr::UsdPrimAllPrimsPredicate));

      size_t num_children = boost::size(children);

      /* Merge only if the Xform has a single Mesh child. */
      if (num_children == 1) {
        pxr::UsdPrim child_prim = children.front();

        if (child_prim && child_prim.GetTypeName() == usdtokens::mesh_type) {
          /* We don't create a reader for the current Xform prim, but instead
           * make a single reader that will merge the Xform and its child. */

          merge_reader = true;
          reader = get_reader(child_prim, context);
          prim_path = child_prim.GetPath().GetString();

          if (reader) {
            reader->set_merged_with_parent(true);
          }
          else {
            std::cerr << "WARNING:  Couldn't get reader when merging child prim." << std::endl;
          }
        }
      }
    }

    if (!merge_reader) {
      reader = get_reader(prim, context);
    }

    if (reader) {
      readers_map.insert(std::make_pair(prim_path, reader));

      /* If we merged, we also add the reader to the map under the parent prim path. */
      if (merge_reader) {
        std::string parent_path = prim.GetPath().GetString();
        if (readers_map.insert(std::make_pair(parent_path, reader)).second == false) {
          std::cerr << "Programmer error: couldn't insert merged prim into reader map with parent "
                       "path key."
                    << std::endl;
        }
      }

      r_readers.push_back(reader);
      reader->incref();
    }
  }

  /* Set parenting. */
  for (UsdObjectReader *r : r_readers) {

    pxr::UsdPrim parent = r->prim().GetParent();

    if (parent && r->merged_with_parent()) {
      /* If we are merging, we use the grandparent. */
      parent = parent.GetParent();
    }

    if (parent) {
      std::string parent_path = parent.GetPath().GetString();

      std::map<std::string, UsdObjectReader *>::const_iterator parent_entry = readers_map.find(
          parent_path);

      if (parent_entry != readers_map.end()) {
        r->set_parent(parent_entry->second);
      }
    }
  }
}

void create_readers(const pxr::UsdPrim &prim,
                    const USDImporterContext &context,
                    std::vector<UsdObjectReader *> &r_readers,
                    std::vector<UsdObjectReader *> &r_child_readers)
{
  if (!prim) {
    return;
  }

  bool is_root = prim.IsPseudoRoot();

  std::vector<UsdObjectReader *> child_readers;

  /* Recursively create readers for the child prims. */
  pxr::UsdPrimSiblingRange child_prims = prim.GetFilteredChildren(
      pxr::UsdTraverseInstanceProxies(pxr::UsdPrimDefaultPredicate));

  for (const pxr::UsdPrim &child_prim : child_prims) {
    create_readers(child_prim, context, r_readers, child_readers);
  }

  if (is_root) {
    /* We're at the pseudo root, so we're done. */
    return;
  }

  /* We prune away empty transform or scope hierarchies (we can add an import flag to make this
   * behavior optional).  Therefore, we skip this prim if it's an Xform or Scope and if
   * it has no corresponding child readers. */
  if ((prim.IsA<pxr::UsdGeomXform>() || prim.IsA<pxr::UsdGeomScope>()) && child_readers.empty()) {
    return;
  }

  /* If this is an Xform prim, see if we can merge with the child reader.
   * We only merge if the child reader hasn't yet been merged
   * and if it corresponds to a mesh prim.  The list of child types that
   * can be merged will be expanded as we support more reader types
   * (e.g., for lights, curves, etc.). */

  if (prim.IsA<pxr::UsdGeomXform>() && child_readers.size() == 1 &&
      !child_readers.front()->merged_with_parent() &&
      child_readers.front()->prim().IsA<pxr::UsdGeomMesh>()) {
    child_readers.front()->set_merged_with_parent(true);
    /* Don't create a reader for the Xform but, instead, return the grandchild
     * that we merged. */
    r_child_readers.push_back(child_readers.front());
    return;
  }

  UsdObjectReader *reader = get_reader(prim, context);

  if (reader) {
    for (UsdObjectReader *child_reader : child_readers) {
      child_reader->set_parent(reader);
    }
    r_child_readers.push_back(reader);
    r_readers.push_back(reader);
  }
  else {
    /* No reader was allocated for this prim, so we pass our child readers back to the caller,
     * for possible handling by a parent reader. */
    r_child_readers.insert(r_child_readers.end(), child_readers.begin(), child_readers.end());
  }
}

} /* namespace blender::io::usd */
