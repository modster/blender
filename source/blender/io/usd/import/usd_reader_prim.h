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
#pragma once

#include "usd.h"
#include "usd_importer_context.h"

#include <pxr/usd/usd/prim.h>

namespace blender::io::usd {

class USDPrimReader {

protected:
  pxr::UsdPrim prim_;

  /* The USD prim path. */
  std::string prim_path_;

  USDImporterContext context_;

  /* Not setting min/max time yet. */
  double min_time_;
  double max_time_;

  /* Use reference counting since the same reader may be used by multiple
  * modifiers and/or constraints. */
  int refcount_;

public:
  explicit USDPrimReader(const pxr::UsdPrim &prim, const USDImporterContext &context);

  virtual ~USDPrimReader();

  const pxr::UsdPrim &prim() const;

  const std::string &prim_path() const
  {
    return prim_path_;
  }

  virtual bool valid() const = 0;

  double min_time() const;
  double max_time() const;

  int refcount() const;
  void incref();
  void decref();
};

} /* namespace blender::io::usd */
