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

#include "usd_data_cache.h"

#include <iostream>

namespace blender::io::usd {

USDDataCache::USDDataCache()
{
}

USDDataCache::~USDDataCache()
{
  /* TODO(makowalski): decrement use count and/or delete the cached data? */
}

bool USDDataCache::add_prototype_mesh(const pxr::SdfPath path, Mesh *mesh)
{
  /* TODO(makowalsk): should we increment mesh use count?  */
  return prototype_meshes_.insert(std::make_pair(path, mesh)).second;
}

void USDDataCache::clear_protype_mesh(const pxr::SdfPath &path)
{
  /* TODO(makowalsk): should we decrement mesh use count or delete mesh?  */
  prototype_meshes_.erase(path);
}

Mesh *USDDataCache::get_prototype_mesh(const pxr::SdfPath &path) const
{
  std::map<pxr::SdfPath, Mesh *>::const_iterator it = prototype_meshes_.find(path);
  if (it != prototype_meshes_.end()) {
    return it->second;
  }

  return nullptr;
}

}  // namespace blender::io::usd
