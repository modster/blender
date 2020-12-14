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

namespace blender::io::usd {

/* Creates a Blender empty object as a placeholder for USD geom
 * sphere, cube, cylinder, cone and capsule schemas. Currently,
 * the viewport display is limited only to the draw modes for
 * empty objects, but is still useful for visualizing transform
 * hierarchies.  Eventually, we could extend this class to
 * derive from USDMeshReaderBase to generate mesh geometry for
 * an accurate representation of the shapes. */

class USDShapeReader : public USDXformableReader {
 protected:
 public:
  USDShapeReader(const pxr::UsdPrim &prim, const USDImporterContext &context);

  virtual ~USDShapeReader();

  bool valid() const override;

  virtual bool can_merge_with_parent() const override
  {
    return true;
  }

  void create_object(Main *bmain, double time, USDDataCache *data_cache) override;
};

}  // namespace blender::io::usd
