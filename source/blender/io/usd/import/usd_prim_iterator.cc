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
#include "usd_data_cache.h"
#include "usd_importer_context.h"
#include "usd_reader_camera.h"
#include "usd_reader_light.h"
#include "usd_reader_mesh.h"
#include "usd_reader_shape.h"
#include "usd_reader_xform.h"
#include "usd_reader_xformable.h"

#include "BLI_listbase.h"
#include "BLI_string.h"
#include "DNA_cachefile_types.h"

#include "MEM_guardedalloc.h"

#include <iostream>
#include <pxr/pxr.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/capsule.h>
#include <pxr/usd/usdGeom/cone.h>
#include <pxr/usd/usdGeom/cube.h>
#include <pxr/usd/usdGeom/cylinder.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/scope.h>
#include <pxr/usd/usdGeom/sphere.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdGeom/xformable.h>
#include <pxr/usd/usdLux/light.h>

namespace blender::io::usd {

USDPrimIterator::USDPrimIterator(pxr::UsdStageRefPtr stage,
                                 const USDImporterContext &context,
                                 Main *bmain)
    : stage_(stage), context_(context), bmain_(bmain)
{
}

USDPrimIterator::USDPrimIterator(const char *file_path,
                                 const USDImporterContext &context,
                                 Main *bmain)
    : stage_(pxr::UsdStage::Open(file_path)), context_(context), bmain_(bmain)
{
}

bool USDPrimIterator::valid() const
{
  return stage_;
}

void USDPrimIterator::create_object_readers(std::vector<USDXformableReader *> &r_readers) const
{
  if (!stage_) {
    return;
  }
  std::vector<USDXformableReader *> child_readers;

  create_object_readers(stage_->GetPseudoRoot(), context_, r_readers, child_readers);
}

void USDPrimIterator::create_prototype_object_readers(
    std::map<pxr::SdfPath, USDXformableReader *> &r_proto_readers) const
{
  if (!stage_) {
    return;
  }

  std::vector<pxr::UsdPrim> protos = stage_->GetMasters();

  for (const pxr::UsdPrim &proto_prim : protos) {
    std::vector<USDXformableReader *> proto_readers;
    std::vector<USDXformableReader *> child_readers;

    create_object_readers(proto_prim, context_, proto_readers, child_readers);

    for (USDXformableReader *reader : proto_readers) {
      if (reader) {
        pxr::UsdPrim reader_prim = reader->prim();
        if (reader_prim) {
          r_proto_readers.insert(std::make_pair(reader_prim.GetPath(), reader));
        }
      }
    }
  }
}

void USDPrimIterator::debug_traverse_stage() const
{
  debug_traverse_stage(stage_);
}

USDXformableReader *USDPrimIterator::get_object_reader(const pxr::UsdPrim &prim)
{
  return get_object_reader(prim, context_);
}

USDXformableReader *USDPrimIterator::get_object_reader(const pxr::UsdPrim &prim,
                                                       const USDImporterContext &context)
{
  USDXformableReader *result = nullptr;

  if (prim.IsA<pxr::UsdGeomCamera>()) {
    result = new USDCameraReader(prim, context);
  }
  else if (prim.IsA<pxr::UsdGeomMesh>()) {
    result = new USDMeshReader(prim, context);
  }
  else if (prim.IsA<pxr::UsdGeomCapsule>() || prim.IsA<pxr::UsdGeomCone>() ||
           prim.IsA<pxr::UsdGeomCube>() || prim.IsA<pxr::UsdGeomCylinder>() ||
           prim.IsA<pxr::UsdGeomSphere>()) {
    result = new USDShapeReader(prim, context);
  }
  else if (prim.IsA<pxr::UsdGeomXform>()) {
    result = new USDXformReader(prim, context);
  }
  else if (prim.IsA<pxr::UsdLuxLight>()) {
    result = new USDLightReader(prim, context);
  }

  return result;
}

void USDPrimIterator::create_object_readers(const pxr::UsdPrim &prim,
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

  int num_child_prims = 0;
  for (const pxr::UsdPrim &child_prim : child_prims) {
    ++num_child_prims;
    create_object_readers(child_prim, context, r_readers, child_readers);
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

  /* If this is an Xform prim, see if we can merge with the child reader. */

  if (prim.IsA<pxr::UsdGeomXform>() && num_child_prims == 1 &&
      !child_readers.front()->merged_with_parent() &&
      child_readers.front()->can_merge_with_parent()) {
    child_readers.front()->set_merged_with_parent(true);
    /* Don't create a reader for the Xform but, instead, return the child
     * that we merged. */
    r_child_readers.push_back(child_readers.front());
    return;
  }

  USDXformableReader *reader = get_object_reader(prim, context);

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

void USDPrimIterator::cache_prototype_data(USDDataCache &r_cache) const
{
  if (!stage_) {
    return;
  }

  std::vector<pxr::UsdPrim> protos = stage_->GetMasters();

  double time = 0.0;

  for (const pxr::UsdPrim &proto_prim : protos) {
    std::vector<USDXformableReader *> proto_readers;
    std::vector<USDXformableReader *> child_readers;

    create_object_readers(proto_prim, context_, proto_readers, child_readers);

    for (USDXformableReader *reader : proto_readers) {
      if (reader) {
        pxr::UsdPrim reader_prim = reader->prim();
        if (reader_prim) {

          if (USDMeshReaderBase *mesh_reader = dynamic_cast<USDMeshReaderBase *>(reader)) {
            Mesh *proto_mesh = mesh_reader->create_mesh(bmain_, time);
            if (proto_mesh) {

              if (this->context_.import_params.import_materials) {
                mesh_reader->assign_materials(this->bmain_, proto_mesh, time, false);
              }

              /* TODO(makowalski): Do we want to decrement the mesh's use count to 0?
               * Might have a small memory leak otherwise. Also, check if mesh is
               * already in cache before adding? */
              r_cache.add_prototype_mesh(reader_prim.GetPath(), proto_mesh);
            }
          }
        }
      }
    }

    /* Clean up the readers. */
    for (USDXformableReader *reader : proto_readers) {
      reader->decref();
    }
  }
}

bool USDPrimIterator::gather_objects_paths(ListBase *object_paths) const
{
  if (!stage_) {
    return false;
  }

  pxr::UsdPrimRange prims = stage_->Traverse(
      pxr::UsdTraverseInstanceProxies(pxr::UsdPrimDefaultPredicate));

  for (const pxr::UsdPrim &prim : prims) {
    void *usd_path_void = MEM_callocN(sizeof(CacheObjectPath), "CacheObjectPath");
    CacheObjectPath *usd_path = static_cast<CacheObjectPath *>(usd_path_void);

    BLI_strncpy(usd_path->path, prim.GetPrimPath().GetString().c_str(), sizeof(usd_path->path));
    BLI_addtail(object_paths, usd_path);
  }

  return true;
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
    if (prim.IsInstanceProxy()) {
      pxr::UsdPrim proto_prim = prim.GetPrimInMaster();
      if (proto_prim) {
        std::cout << "  Prototype prim: " << proto_prim.GetPath() << std::endl;
      }
    }
  }
}

} /* namespace blender::io::usd */
