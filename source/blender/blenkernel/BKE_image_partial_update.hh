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
 * The Original Code is Copyright (C) 2001-2002 by NaN Holding BV.
 * All rights reserved.
 */
#pragma once

/** \file
 * \ingroup bke
 */

#include "DNA_image_types.h"

#include "BKE_image.h"

namespace blender::bke::image {

class PartialUpdateIterator {
  Image *image;
  PartialUpdateUser *user;

 public:
  int tile_number;
  ImageTile *tile = nullptr;
  ImBuf *tile_buffer = nullptr;
  PartialUpdateRegion change;

 public:
  PartialUpdateIterator(Image *image, PartialUpdateUser *user)
      : image(image), user(user), tile_number(-1)
  {
  }

  ePartialUpdateCollectResult collect_changes()
  {
    return BKE_image_partial_update_collect_changes(image, user);
  }

  ePartialUpdateIterResult get_next_change()
  {
    ePartialUpdateIterResult result = BKE_image_partial_update_get_next_change(user, &change);
    switch (result) {
      case PARTIAL_UPDATE_ITER_FINISHED:
        free_tile_buffer();
        return result;

      case PARTIAL_UPDATE_ITER_CHANGE_AVAILABLE:
        if (tile_number == change.tile_number) {
          return result;
        }
        free_tile_buffer();
        acquire_tile_buffer(change.tile_number);
        return result;

      default:
        BLI_assert_unreachable();
        return result;
    }
  }

 private:
  void free_tile_buffer()
  {
    BKE_image_release_ibuf(image, tile_buffer, nullptr);
    tile = nullptr;
    tile_buffer = nullptr;
  }

  void acquire_tile_buffer(int new_tile_number)
  {
    ImageUser tile_user = {0};
    tile_user.tile = new_tile_number;

    tile = BKE_image_get_tile(image, tile_number);
    tile_buffer = BKE_image_acquire_ibuf(image, &tile_user, NULL);
    tile_number = new_tile_number;
  }
};

}  // namespace blender::bke::image
