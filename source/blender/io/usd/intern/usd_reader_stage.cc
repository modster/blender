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
#include "usd_reader_mesh.h"
#include "usd_reader_prim.h"
#include "usd_reader_xform.h"

#include "usd_util.h"

extern "C" {
#include "DEG_depsgraph.h"
#include "DEG_depsgraph_build.h"
#include "DEG_depsgraph_query.h"

#include "DNA_node_types.h"
#include "DNA_scene_types.h"
#include "DNA_world_types.h"

#include "BKE_blender_version.h"
#include "BKE_context.h"
#include "BKE_global.h"
#include "BKE_main.h"
#include "BKE_node.h"
#include "BKE_scene.h"
#include "BKE_world.h"

#include "BLI_fileops.h"
#include "BLI_path_util.h"
#include "BLI_string.h"

#include "WM_api.h"
#include "WM_types.h"
}
#include <iostream>

namespace blender::io::usd {

USDStageReader::USDStageReader(struct Main *bmain, const char *filename)
{
  m_stage = pxr::UsdStage::Open(filename);
}

USDStageReader::~USDStageReader()
{
  clear_readers();

  m_stage->Unload();
}

bool USDStageReader::valid() const
{
  // TODO: Implement
  return true;
}


/* Returns true if the given prim should be excluded from the
 * traversal because it's invisible. */
bool _prune_by_visibility(const pxr::UsdGeomImageable &imageable,
                          const USDImportParams &params)
{
  if (imageable && params.import_visible_only) {
    if (pxr::UsdAttribute visibility_attr = imageable.GetVisibilityAttr()) {

      // Prune if the prim has a non-animating visibility attribute and is
      // invisible.
      if (!visibility_attr.ValueMightBeTimeVarying()) {
        pxr::TfToken visibility;
        visibility_attr.Get(&visibility);
        return visibility == pxr::UsdGeomTokens->invisible;
      }
    }
  }

  return false;
}


/* Returns true if the given prim should be excluded from the
 * traversal because it has a purpose which was not requested
 * by the user; e.g., the prim represents guide geometry and
 * the import_guide parameter is toggled off. */
bool _prune_by_purpose(const pxr::UsdGeomImageable &imageable,
                       const USDImportParams &params)
{
  if (imageable && !(params.import_guide && params.import_proxy && params.import_render)) {
    if (pxr::UsdAttribute purpose_attr = imageable.GetPurposeAttr()) {
      pxr::TfToken purpose;
      purpose_attr.Get(&purpose);
      if ((!params.import_guide && purpose == pxr::UsdGeomTokens->guide) ||
        (!params.import_proxy && purpose == pxr::UsdGeomTokens->proxy) ||
        (!params.import_render && purpose == pxr::UsdGeomTokens->render)) {
        return true;
      }
    }
  }

  return false;
}

static USDPrimReader *_handlePrim(Main *bmain,
                                  pxr::UsdStageRefPtr stage,
                                  const USDImportParams &params,
                                  pxr::UsdPrim prim,
                                  USDPrimReader *parent_reader,
                                  std::vector<USDPrimReader *> &readers,
                                  ImportSettings &settings)
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

  USDPrimReader *reader = NULL;

  // This check prevents the pseudo 'root' prim to be added
  if (prim != stage->GetPseudoRoot()) {
    reader = blender::io::usd::create_reader(stage, prim, params, settings);
    if (reader == NULL)
      return NULL;

    reader->parent(parent_reader);
    reader->createObject(bmain, 0.0);

    readers.push_back(reader);
    reader->incref();
  }

  pxr::Usd_PrimFlagsPredicate filter_predicate = pxr::UsdPrimDefaultPredicate;

  if (params.import_instance_proxies) {
    filter_predicate = pxr::UsdTraverseInstanceProxies(filter_predicate);
  }

  pxr::UsdPrimSiblingRange children = prim.GetFilteredChildren(filter_predicate);

  for (const auto &childPrim : children) {
    _handlePrim(bmain, stage, params, childPrim, reader, readers, settings);
  }

  return reader;
}

std::vector<USDPrimReader *> USDStageReader::collect_readers(Main *bmain,
                                                             const USDImportParams &params,
                                                             ImportSettings &settings)
{
  m_params = params;
  m_settings = settings;

  clear_readers();

  // Iterate through stage
  pxr::UsdPrim root = m_stage->GetPseudoRoot();

  std::string prim_path_mask(params.prim_path_mask);

  if (prim_path_mask.size() > 0) {
    std::cout << prim_path_mask << '\n';
    pxr::SdfPath path = pxr::SdfPath(prim_path_mask);
    pxr::UsdPrim prim = m_stage->GetPrimAtPath(path.StripAllVariantSelections());
    if (prim.IsValid()) {
      root = prim;
      if (path.ContainsPrimVariantSelection()) {
        // TODO: This will not work properly with setting variants on child prims
        while (path.ContainsPrimVariantSelection()) {
          std::pair<std::string, std::string> variantSelection = path.GetVariantSelection();
          root.GetVariantSet(variantSelection.first).SetVariantSelection(variantSelection.second);
          path = path.GetParentPath();
        }
      }
    }
  }

  m_stage->SetInterpolationType(pxr::UsdInterpolationType::UsdInterpolationTypeHeld);
  _handlePrim(bmain, m_stage, params, root, NULL, m_readers, settings);

  return m_readers;
}

void USDStageReader::clear_readers()
{
  std::vector<USDPrimReader *>::iterator iter;
  for (iter = m_readers.begin(); iter != m_readers.end(); ++iter) {
    if (((USDPrimReader *)*iter)->refcount() == 0) {
      delete *iter;
    }
  }
}

}  // Namespace blender::io::usd

