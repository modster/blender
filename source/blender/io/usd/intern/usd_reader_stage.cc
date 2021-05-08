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
 * along with this program; if not, write to the Free Software  Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 * The Original Code is Copyright (C) 2016 Kévin Dietrich.
 * All rights reserved.
 */

#include "usd_reader_stage.h"
#include "usd_reader_camera.h"
#include "usd_reader_curve.h"
#include "usd_reader_instance.h"
#include "usd_reader_light.h"
#include "usd_reader_mesh.h"
#include "usd_reader_nurbs.h"
#include "usd_reader_prim.h"
#include "usd_reader_volume.h"
#include "usd_reader_xform.h"

#include <pxr/pxr.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/curves.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/nurbsCurves.h>
#include <pxr/usd/usdGeom/scope.h>
#include <pxr/usd/usdLux/light.h>

#include <iostream>

namespace blender::io::usd {

USDStageReader::USDStageReader(const char *filename)
{
  stage_ = pxr::UsdStage::Open(filename);
}

USDStageReader::~USDStageReader()
{
  clear_readers(true);

  if (stage_) {
    stage_->Unload();
  }
}

bool USDStageReader::valid() const
{
  return stage_;
}

USDPrimReader *USDStageReader::create_reader(const pxr::UsdPrim &prim,
                                             const USDImportParams &params,
                                             const ImportSettings &settings)
{
  USDPrimReader *reader = nullptr;

  if (params.import_cameras && prim.IsA<pxr::UsdGeomCamera>()) {
    reader = new USDCameraReader(prim, params, settings);
  }
  else if (params.import_curves && prim.IsA<pxr::UsdGeomBasisCurves>()) {
    reader = new USDCurvesReader(prim, params, settings);
  }
  else if (params.import_curves && prim.IsA<pxr::UsdGeomNurbsCurves>()) {
    reader = new USDNurbsReader(prim, params, settings);
  }
  else if (params.import_meshes && prim.IsA<pxr::UsdGeomMesh>()) {
    reader = new USDMeshReader(prim, params, settings);
  }
  else if (params.import_lights && prim.IsA<pxr::UsdLuxLight>()) {
    reader = new USDLightReader(prim, params, settings);
  }
  else if (params.import_volumes && prim.IsA<pxr::UsdVolVolume>()) {
    reader = new USDVolumeReader(prim, params, settings);
  }
  else if (prim.IsA<pxr::UsdGeomImageable>()) {
    reader = new USDXformReader(prim, params, settings);
  }

  return reader;
}

// TODO(makowalski): The handle does not have the proper import params or settings
USDPrimReader *USDStageReader::create_reader(const USDStageReader *archive,
                                             const pxr::UsdPrim &prim)
{
  USDPrimReader *reader = nullptr;

  if (prim.IsA<pxr::UsdGeomCamera>()) {
    reader = new USDCameraReader(prim, archive->params(), archive->settings());
  }
  else if (prim.IsA<pxr::UsdGeomBasisCurves>()) {
    reader = new USDCurvesReader(prim, archive->params(), archive->settings());
  }
  else if (prim.IsA<pxr::UsdGeomNurbsCurves>()) {
    reader = new USDNurbsReader(prim, archive->params(), archive->settings());
  }
  else if (prim.IsA<pxr::UsdGeomMesh>()) {
    reader = new USDMeshReader(prim, archive->params(), archive->settings());
  }
  else if (prim.IsA<pxr::UsdLuxLight>()) {
    reader = new USDLightReader(prim, archive->params(), archive->settings());
  }
  else if (prim.IsA<pxr::UsdVolVolume>()) {
    reader = new USDVolumeReader(prim, archive->params(), archive->settings());
  }
  else if (prim.IsA<pxr::UsdGeomImageable>()) {
    reader = new USDXformReader(prim, archive->params(), archive->settings());
  }
  return reader;
}

/* Returns true if the given prim should be excluded from the
 * traversal because it's invisible. */
static bool _prune_by_visibility(const pxr::UsdGeomImageable &imageable,
                                 const USDImportParams &params)
{
  if (!(imageable && params.import_visible_only)) {
    return false;
  }

  if (pxr::UsdAttribute visibility_attr = imageable.GetVisibilityAttr()) {
    // Prune if the prim has a non-animating visibility attribute and is
    // invisible.
    if (!visibility_attr.ValueMightBeTimeVarying()) {
      pxr::TfToken visibility;
      visibility_attr.Get(&visibility);
      return visibility == pxr::UsdGeomTokens->invisible;
    }
  }

  return false;
}

/* Returns true if the given prim should be excluded from the
 * traversal because it has a purpose which was not requested
 * by the user; e.g., the prim represents guide geometry and
 * the import_guide parameter is toggled off. */
static bool _prune_by_purpose(const pxr::UsdGeomImageable &imageable,
                              const USDImportParams &params)
{
  if (!imageable) {
    return false;
  }

  if (params.import_guide && params.import_proxy && params.import_render) {
    return false;
  }

  if (pxr::UsdAttribute purpose_attr = imageable.GetPurposeAttr()) {
    pxr::TfToken purpose;
    if (!purpose_attr.Get(&purpose)) {
      return false;
    }
    if (purpose == pxr::UsdGeomTokens->guide && !params.import_guide) {
      return true;
    }
    if (purpose == pxr::UsdGeomTokens->proxy && !params.import_proxy) {
      return true;
    }
    if (purpose == pxr::UsdGeomTokens->render && !params.import_render) {
      return true;
    }
  }

  return false;
}

/* Determine if the given reader can use the parent of the encapsulated USD prim
 * to compute the Blender object's transform. If so, the reader is appropriately
 * flagged and the function returns true. Otherwise, the function returns false. */
static bool _merge_with_parent(USDPrimReader *reader)
{
  USDXformReader *xform_reader = dynamic_cast<USDXformReader *>(reader);

  if (!xform_reader) {
    return false;
  }

  /* Check if the Xform reader is already merged. */
  if (xform_reader->use_parent_xform()) {
    return false;
  }

  /* Only merge if the parent is an Xform. */
  if (!xform_reader->prim().GetParent().IsA<pxr::UsdGeomXform>()) {
    return false;
  }

  /* Don't merge Xform and Scope prims. */
  if (xform_reader->prim().IsA<pxr::UsdGeomXform>() ||
      xform_reader->prim().IsA<pxr::UsdGeomScope>()) {
    return false;
  }

  /* Don't merge if the prim has authored transform ops. */
  if (xform_reader->prim_has_xform_ops()) {
    return false;
  }

  /* Flag the Xform reader as merged. */
  xform_reader->set_use_parent_xform(true);

  return true;
}

static USDPrimReader *_handlePrim(Main *bmain,
                                  pxr::UsdStageRefPtr stage,
                                  const USDImportParams &params,
                                  pxr::UsdPrim prim,
                                  std::vector<USDPrimReader *> &readers,
                                  const ImportSettings &settings)
{
  if (prim.IsA<pxr::UsdGeomImageable>()) {
    pxr::UsdGeomImageable imageable(prim);

    if (_prune_by_purpose(imageable, params)) {
      return nullptr;
    }

    if (_prune_by_visibility(imageable, params)) {
      return nullptr;
    }
  }

  pxr::Usd_PrimFlagsPredicate filter_predicate = pxr::UsdPrimDefaultPredicate;

  if (params.import_instance_proxies) {
    filter_predicate = pxr::UsdTraverseInstanceProxies(filter_predicate);
  }

  pxr::UsdPrimSiblingRange children = prim.GetFilteredChildren(filter_predicate);

  std::vector<USDPrimReader *> child_readers;

  for (const auto &childPrim : children) {
    USDPrimReader *child_reader = _handlePrim(bmain, stage, params, childPrim, readers, settings);
    if (child_reader) {
      child_readers.push_back(child_reader);
    }
  }

  /* Check if we can merge an Xform with its child prim. */
  if (child_readers.size() == 1) {

    USDPrimReader *child_reader = child_readers.front();

    if (_merge_with_parent(child_reader)) {
      return child_reader;
    }
  }

  USDPrimReader *reader = nullptr;

  if (!prim.IsPseudoRoot()) {
    reader = USDStageReader::create_reader(prim, params, settings);

    if (reader == nullptr) {
      return nullptr;
    }

    reader->create_object(bmain, 0.0);

    readers.push_back(reader);
    reader->incref();

    /* Set each child reader's parent. */
    for (USDPrimReader *child_reader : child_readers) {
      child_reader->parent(reader);
    }
  }

  return reader;
}

void USDStageReader::collect_readers(Main *bmain,
                                     const USDImportParams &params,
                                     const ImportSettings &settings)
{
  params_ = params;
  settings_ = settings;

  clear_readers(true);

  // Iterate through stage
  pxr::UsdPrim root = stage_->GetPseudoRoot();

  std::string prim_path_mask(params.prim_path_mask);

  if (prim_path_mask.size() > 0) {
    std::cout << prim_path_mask << '\n';
    pxr::SdfPath path = pxr::SdfPath(prim_path_mask);
    pxr::UsdPrim prim = stage_->GetPrimAtPath(path.StripAllVariantSelections());
    if (prim.IsValid()) {
      root = prim;
      if (path.ContainsPrimVariantSelection()) {
        // TODO(makowalski): This will not work properly with setting variants on child prims
        while (path.ContainsPrimVariantSelection()) {
          std::pair<std::string, std::string> variantSelection = path.GetVariantSelection();
          root.GetVariantSet(variantSelection.first).SetVariantSelection(variantSelection.second);
          path = path.GetParentPath();
        }
      }
    }
  }

  stage_->SetInterpolationType(pxr::UsdInterpolationType::UsdInterpolationTypeHeld);
  _handlePrim(bmain, stage_, params, root, readers_, settings);
}

void USDStageReader::clear_readers(bool decref)
{
  for (USDPrimReader *reader : readers_) {
    if (!reader) {
      continue;
    }

    if (decref) {
      reader->decref();
    }

    if (reader->refcount() == 0) {
      delete reader;
    }
  }

  readers_.clear();
}

}  // Namespace blender::io::usd
