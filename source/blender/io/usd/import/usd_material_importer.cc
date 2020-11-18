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

#include "usd_material_importer.h"

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"
#include "DNA_object_types.h"

#include "BKE_lib_id.h"
#include "BKE_main.h"
#include "BKE_material.h"
#include "BKE_mesh.h"
#include "BKE_object.h"



#include <iostream>

namespace blender::io::usd {

USDMaterialImporter::USDMaterialImporter(const USDImporterContext &context, Main *bmain)
  : context_(context)
  , bmain_(bmain)
{
}

USDMaterialImporter::~USDMaterialImporter()
{
}


Material *USDMaterialImporter::add_material(const pxr::UsdShadeMaterial &usd_material) const
{
  if (!(bmain_ && usd_material)) {
    return nullptr;
  }

  std::string mtl_name = usd_material.GetPrim().GetName().GetString().c_str();

  return BKE_material_add(bmain_, mtl_name.c_str());
}

}  // namespace blender::io::usd
