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
 * Copyright 2021, Blender Foundation.
 */

/** \file
 * \ingroup eevee
 *
 * A cluster object is a data structure that contains fine grained culling
 * of entities against in the whole view frustum.
 */

#include "DRW_render.h"

#include "eevee_shader_shared.hh"

namespace blender::eevee {

class Cluster : public ClusterDataBuf {
 private:
  const DRWView *view;
  float viewvecs[2][4];

 public:
  Cluster(){};
  ~Cluster(){};

  void set_view(const DRWView *view, const int UNUSED(extent[2]))
  {
    /* Reset bitarray. */
    ClusterDataBuf *data = static_cast<ClusterDataBuf *>(this);
    memset(data->cells, 0x0, sizeof(data->cells));
  }

  /* Inject the object in the cluster. */
  void insert(BoundSphere &UNUSED(bsphere), uint64_t index)
  {
    /* Current limitation of using a uvec4 (128 flat bit array). */
    BLI_assert(index < 128);

    set_index_bit_enabled(0, index);
  }

 private:
  void set_index_bit_enabled(uint64_t cell_index, uint64_t item_index)
  {
    if (item_index > 63) {
      *((uint64_t *)&cells[cell_index][2]) |= 1lu << (item_index - 64);
    }
    else {
      *((uint64_t *)&cells[cell_index][0]) |= 1lu << item_index;
    }
  }
};

}  // namespace blender::eevee