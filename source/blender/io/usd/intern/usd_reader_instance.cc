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
 * The Original Code is Copyright (C) 2021 Blender Foundation.
 * All rights reserved.
 */

#include "usd_reader_instance.h"

#include "BKE_object.h"
#include "DNA_object_types.h"

#include <iostream>

namespace blender::io::usd {

USDInstanceReader::USDInstanceReader(pxr::UsdStageRefPtr stage,
                                     const pxr::UsdPrim &object,
                                     const USDImportParams &import_params,
                                     ImportSettings &settings)
    : USDXformReader(stage, object, import_params, settings)
{
}

bool USDInstanceReader::valid() const
{
  return m_prim.IsValid() && m_prim.IsInstance();
}

void USDInstanceReader::createObject(Main *bmain, double motionSampleTime)
{
  this->m_object = BKE_object_add_only_object(bmain, OB_EMPTY, m_name.c_str());
  this->m_object->data = nullptr;
  this->m_object->transflag |= OB_DUPLICOLLECTION;
}

void USDInstanceReader::set_instance_collection(Collection *coll)
{
  if (coll && this->m_object) {
    this->m_object->instance_collection = coll;
  }
}

pxr::SdfPath USDInstanceReader::proto_path() const
{
  if (pxr::UsdPrim master = m_prim.GetMaster()) {
    return master.GetPath();
  }

  return pxr::SdfPath();
}

}  // namespace blender::io::usd
