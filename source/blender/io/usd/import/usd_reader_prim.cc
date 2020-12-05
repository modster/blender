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

#include "usd_reader_prim.h"
#include "BLI_assert.h"

#include <iostream>

namespace blender::io::usd {

USDPrimReader::USDPrimReader(const pxr::UsdPrim &prim, const USDImporterContext &context)
    : prim_(prim),
      prim_path_(""),
      context_(context),
      min_time_(std::numeric_limits<double>::max()),
      max_time_(std::numeric_limits<double>::min()),
      refcount_(1)
{
  if (prim) {
    prim_path_ = prim.GetPath().GetString();
  }
}

USDPrimReader::~USDPrimReader()
{
}

const pxr::UsdPrim &USDPrimReader::prim() const
{
  return prim_;
}

double USDPrimReader::min_time() const
{
  return min_time_;
}

double USDPrimReader::max_time() const
{
  return max_time_;
}

int USDPrimReader::incref()
{
  return ++refcount_;
}

int USDPrimReader::decref()
{
  --refcount_;
  BLI_assert(refcount_ >= 0);
  if (refcount_ == 0) {
    delete this;
    return 0;
  }
  return refcount_;
}

} /* namespace blender::io::usd */
