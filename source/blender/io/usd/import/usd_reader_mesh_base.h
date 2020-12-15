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

#include <map>

struct Mesh;

namespace blender::io::usd {

/* Abstract base class of readers that can create a Blender mesh object.
 * Subclasses must define the create_mesh() and assign_materials() virtual
 * functions to implement the logic for creating the Mesh data and materials. */

class USDMeshReaderBase : public USDXformableReader {
 protected:
 public:
  USDMeshReaderBase(const pxr::UsdPrim &prim, const USDImporterContext &context);

  virtual ~USDMeshReaderBase();

  void create_object(Main *bmain, double time, USDDataCache *data_cache) override;

  struct Mesh *read_mesh(Main *bmain, double time, USDDataCache *data_cache, bool &r_is_instance);

  virtual struct Mesh *create_mesh(Main *bmain, double time) = 0;

  /* If mesh isn't null, assign material indices to the mesh faces.
   * If set_object_materials is true, assign materials to the object. */
  virtual void assign_materials(Main *bmain,
                                Mesh *mesh,
                                double time,
                                bool set_object_materials) = 0;
};

}  // namespace blender::io::usd
