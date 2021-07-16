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
 * The Original Code is Copyright (C) Blender Foundation
 * All rights reserved.
 */

/** \file
 * \ingroup bke
 */

#include "DNA_cloth_types.h"
#include "DNA_mesh_types.h"
#include "DNA_object_types.h"

#include "BLI_utildefines.h"

#include "BKE_cloth.h"
#include "BKE_cloth_remesh.hh"

#include <cstdio>

namespace blender::bke {

Mesh *BKE_cloth_remesh(Object *ob, ClothModifierData *clmd, Mesh *mesh)
{
  auto *cloth_to_object_res = cloth_to_object(ob, clmd, mesh, false);
  BLI_assert(cloth_to_object_res == nullptr);

  internal::MeshIO meshio;

  meshio.read(mesh);

  return meshio.write();
}

}  // namespace blender::bke
