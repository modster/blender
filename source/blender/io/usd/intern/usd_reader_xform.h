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
 */
#pragma once

#include "usd.h"
#include "usd_reader_prim.h"

namespace blender::io::usd {

class USDXformReader : public USDPrimReader {

 public:
  USDXformReader(pxr::UsdStageRefPtr stage,
                 const pxr::UsdPrim &object,
                 const USDImportParams &import_params,
                 ImportSettings &settings)
      : USDPrimReader(stage, object, import_params, settings)
  {
  }

  void create_object(Main *bmain, double motionSampleTime) override;
  void read_object_data(Main *bmain, double motionSampleTime) override;

  void read_matrix(float r_mat[4][4], const float time, const float scale, bool &is_constant);

  // Returns true if this reader represents an object that is the root of the
  // transform hierarchy.
  bool is_root_xform_object() const;
};

}  // namespace blender::io::usd
