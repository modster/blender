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

#include "usd_reader_mesh_base.h"
#include "usd_data_cache.h"

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"
#include "DNA_object_types.h"

#include "BKE_lib_id.h"
#include "BKE_main.h"
#include "BKE_mesh.h"
#include "BKE_object.h"

#include <pxr/usd/usdGeom/imageable.h>

#include <iostream>

namespace blender::io::usd {

USDMeshReaderBase::USDMeshReaderBase(const pxr::UsdPrim &prim, const USDImporterContext &context)
    : USDXformableReader(prim, context)
{
}

USDMeshReaderBase::~USDMeshReaderBase()
{
}

void USDMeshReaderBase::create_object(Main *bmain, double time, USDDataCache *data_cache)
{
  if (!this->valid()) {
    return;
  }

  /* Determine mesh visibility.
   * TODO(makowalski): Consider optimizations to avoid this expensive call,
   * for example, by determining visibility during stage traversal. */

  pxr::UsdGeomImageable imageable(this->prim_);

  if (!imageable) {
    std::cerr << "Warning:  Invalid mesh imageable schema for " << this->prim_path_ << std::endl;
    return;
  }

  pxr::TfToken vis_tok = imageable.ComputeVisibility();

  if (vis_tok == pxr::UsdGeomTokens->invisible) {
    return;
  }

  std::string obj_name = get_object_name();

  if (obj_name.empty()) {
    /* Sanity check. */
    std::cerr << "Warning: couldn't determine object name for " << this->prim_path() << std::endl;
  }

  object_ = BKE_object_add_only_object(bmain, OB_MESH, obj_name.c_str());

  bool is_mesh_instance = false;
  Mesh *mesh = this->read_mesh(bmain, time, data_cache, is_mesh_instance);
  object_->data = mesh;

  if (this->context_.import_params.import_materials) {
    assign_materials(bmain, (is_mesh_instance ? nullptr : mesh), time, true);
  }
}

Mesh *USDMeshReaderBase::read_mesh(Main *bmain,
                                   double time,
                                   USDDataCache *data_cache,
                                   bool &r_is_instance)
{
  r_is_instance = false;
  return create_mesh(bmain, time);
}

}  // namespace blender::io::usd
