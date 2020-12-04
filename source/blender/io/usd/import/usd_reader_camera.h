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

#include "usd_reader_xformable.h"

#include <pxr/usd/usdGeom/camera.h>

namespace blender::io::usd {

/* Wraps the UsdGeomCamera schema. Creates a Blender Camera object. */

class USDCameraReader : public USDXformableReader {

  pxr::UsdGeomCamera camera_;

 public:
  USDCameraReader(const pxr::UsdPrim &prim, const USDImporterContext &context);

  bool valid() const override;

  void create_object(Main *bmain, double time, USDDataCache *data_cache) override;

  void read_matrix(float r_mat[4][4], const double time, const float scale) const override;
};

}  // namespace blender::io::usd
