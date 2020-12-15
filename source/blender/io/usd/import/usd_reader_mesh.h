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

#include "usd_reader_mesh_base.h"

#include <pxr/usd/usdGeom/mesh.h>

namespace blender::io::usd {

/* Wraps the UsdGeomMesh schema. Defines the logic for reading
 * Mesh data for the object created by USDMeshReaderBase. */

class USDMeshReader : public USDMeshReaderBase {
 protected:
  pxr::UsdGeomMesh mesh_;

 public:
  USDMeshReader(const pxr::UsdPrim &prim, const USDImporterContext &context);

  virtual ~USDMeshReader();

  bool valid() const override;

  struct Mesh *create_mesh(Main *bmain, double time) override;
  void assign_materials(Main *bmain, Mesh *mesh, double time, bool set_object_materials) override;
};

}  // namespace blender::io::usd
