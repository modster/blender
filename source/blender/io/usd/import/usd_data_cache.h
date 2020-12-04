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

#include <map>

#include <pxr/usd/sdf/path.h>

struct Mesh;

namespace blender::io::usd {

/* Caches data imported from USD, typically shared data for instanced primitives. */

class USDDataCache {
 protected:
  /* Shared meshes for instancing. */
  std::map<pxr::SdfPath, Mesh *> prototype_meshes_;

 public:
  USDDataCache();

  ~USDDataCache();

  const std::map<pxr::SdfPath, Mesh *> &prototype_meshes() const
  {
    return prototype_meshes_;
  }

  void clear_prototype_meshes()
  {
    /* TODO(makowalsk): should we decrement mesh use counts or delete meshes?  */
    prototype_meshes_.clear();
  }

  bool add_prototype_mesh(const pxr::SdfPath path, Mesh *mesh);

  void clear_protype_mesh(const pxr::SdfPath &path);

  Mesh *get_prototype_mesh(const pxr::SdfPath &path) const;

};

}  // namespace blender::io::usd
