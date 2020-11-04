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

#include "usd_prim_iterator.h"

#include "usd.h"
#include "usd_importer_context.h"
#include "usd_reader_mesh.h"
#include "usd_reader_object.h"
#include "usd_reader_transform.h"

#include <iostream>
#include <pxr/base/plug/registry.h>
#include <pxr/pxr.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/scope.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdGeom/xformable.h>

namespace blender::io::usd {

USDXformableReader *USDPrimIterator::get_reader(const pxr::UsdPrim &prim,
                                                const USDImporterContext &context)
{
  USDXformableReader *result = nullptr;

  if (prim.IsA<pxr::UsdGeomMesh>()) {
    result = new USDMeshReader(prim, context);
  }
  else if (prim.IsA<pxr::UsdGeomXform>()) {
    result = new USDTransformReader(prim, context);
  }

  return result;
}

void USDPrimIterator::create_readers(const pxr::UsdPrim &prim,
                                     const USDImporterContext &context,
                                     std::vector<USDXformableReader *> &r_readers,
                                     std::vector<USDXformableReader *> &r_child_readers)
{
  if (!prim) {
    return;
  }

  std::vector<USDXformableReader *> child_readers;

  /* Recursively create readers for the child prims. */
  pxr::UsdPrimSiblingRange child_prims = prim.GetFilteredChildren(
      pxr::UsdTraverseInstanceProxies(pxr::UsdPrimDefaultPredicate));

  for (const pxr::UsdPrim &child_prim : child_prims) {
    create_readers(child_prim, context, r_readers, child_readers);
  }

  if (prim.IsPseudoRoot()) {
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
    /* Don't create a reader for the Xform but, instead, return the child
     * that we merged. */
    r_child_readers.push_back(child_readers.front());
    return;
  }

  USDXformableReader *reader = get_reader(prim, context);

  if (reader) {
    for (USDXformableReader *child_reader : child_readers) {
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

void USDPrimIterator::debug_traverse_stage(const pxr::UsdStageRefPtr &usd_stage)
{
  if (!usd_stage) {
    return;
  }

  pxr::UsdPrimRange prims = usd_stage->Traverse(
      pxr::UsdTraverseInstanceProxies(pxr::UsdPrimAllPrimsPredicate));

  for (const pxr::UsdPrim &prim : prims) {
    std::cout << prim.GetPath() << std::endl;
    std::cout << "  Type: " << prim.GetTypeName() << std::endl;
    ;
  }
}

} /* namespace blender::io::usd */
