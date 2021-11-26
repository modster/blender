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
 * \ingroup bke
 */

#pragma once

#include "BLI_utildefines.h"

#include "BLI_rect.h"

#include "DNA_image_types.h"

extern "C" {
struct PartialUpdateUser;
struct PartialUpdateRegister;
}

namespace blender::bke::image {
  
using TileNumber = int;

namespace partial_update {

/* --- image_partial_update.cc --- */
/** Image partial updates. */

/**
 * \brief Result codes of #BKE_image_partial_update_collect_changes.
 */
typedef enum ePartialUpdateCollectResult {
  /** \brief Unable to construct partial updates. Caller should perform a full update. */
  PARTIAL_UPDATE_NEED_FULL_UPDATE,

  /** \brief No changes detected since the last time requested. */
  PARTIAL_UPDATE_NO_CHANGES,

  /** \brief Changes detected since the last time requested. */
  PARTIAL_UPDATE_CHANGES_AVAILABLE,
} ePartialUpdateCollectResult;

/**
 * \brief A region to update.
 *
 * Data is organized in tiles. These tiles are in texel space (1 unit is a single texel). When
 * tiles are requested they are merged with neighboring tiles.
 */
struct PartialUpdateRegion {
  /** \brief region of the image that has been updated. Region can be bigger than actual changes.
   */
  struct rcti region;

  /**
   * \brief Tile number (UDIM) that this region belongs to.
   */
  TileNumber tile_number;
};

/**
 * \brief Return codes of #BKE_image_partial_update_get_next_change.
 */
typedef enum ePartialUpdateIterResult {
  /** \brief no tiles left when iterating over tiles. */
  PARTIAL_UPDATE_ITER_FINISHED = 0,

  /** \brief a chunk was available and has been loaded. */
  PARTIAL_UPDATE_ITER_CHANGE_AVAILABLE = 1,
} ePartialUpdateIterResult;

/**
 * \brief collect the partial update since the last request.
 *
 * Invoke #BKE_image_partial_update_get_next_change to iterate over the collected tiles.
 *
 * \returns PARTIAL_UPDATE_NEED_FULL_UPDATE: called should not use partial updates but
 *              recalculate the full image. This result can be expected when called
 *              for the first time for a user and when it isn't possible to reconstruct
 *              the changes as the internal state doesn't have enough data stored.
 *          PARTIAL_UPDATE_NO_CHANGES: The have been no changes detected since last
 *              invoke for the same user.
 *          PARTIAL_UPDATE_CHANGES_AVAILABLE: Parts of the image has been updated
 *              since last invoke for the same user. The changes can be read by
 *              using #BKE_image_partial_update_get_next_change.
 */
ePartialUpdateCollectResult BKE_image_partial_update_collect_changes(
    struct Image *image, struct PartialUpdateUser *user);

ePartialUpdateIterResult BKE_image_partial_update_get_next_change(
    struct PartialUpdateUser *user, struct PartialUpdateRegion *r_region);

class AbstractTileData {
 protected:
  virtual ~AbstractTileData() = default;

 public:
  virtual void init_data(TileNumber tile_number) = 0;
  virtual void free_data() = 0;
};

class ImageTileData : AbstractTileData {
 public:
  ImageTile *tile = nullptr;
  ImBuf *tile_buffer = nullptr;

  Image *image;
  ImageUser image_user;

  ImageTileData(Image *image, ImageUser image_user) : image(image), image_user(image_user)
  {
  }

  void init_data(TileNumber new_tile_number) override
  {
    image_user.tile = new_tile_number;

    tile = BKE_image_get_tile(image, new_tile_number);
    tile_buffer = BKE_image_acquire_ibuf(image, &image_user, NULL);
  }

  void free_data() override
  {
    BKE_image_release_ibuf(image, tile_buffer, nullptr);
    tile = nullptr;
    tile_buffer = nullptr;
  }
};

class NoTileData : AbstractTileData {
 public:
  NoTileData(Image *UNUSED(image), ImageUser &UNUSED(image_user))
  {
  }

  void init_data(TileNumber UNUSED(new_tile_number)) override
  {
  }

  void free_data() override
  {
  }
};

template<typename TileData = NoTileData> struct PartialUpdateCollectResult {
  Image *image;
  PartialUpdateUser *user;
  TileData tile_data;
  TileNumber last_tile_number;
  PartialUpdateRegion changed_region;

 private:
  ePartialUpdateCollectResult collect_result;

 public:
  PartialUpdateCollectResult(Image *image,
                             ImageUser image_user,
                             PartialUpdateUser *user,
                             ePartialUpdateCollectResult collect_result)
      : image(image), user(user), tile_data(image, image_user), collect_result(collect_result)
  {
  }

  const ePartialUpdateCollectResult get_collect_result() const
  {
    return collect_result;
  }

  ePartialUpdateIterResult get_next_change()
  {
    BLI_assert(collect_result == PARTIAL_UPDATE_CHANGES_AVAILABLE);
    ePartialUpdateIterResult result = BKE_image_partial_update_get_next_change(user,
                                                                               &changed_region);
    switch (result) {
      case PARTIAL_UPDATE_ITER_FINISHED:
        tile_data.free_data();
        return result;

      case PARTIAL_UPDATE_ITER_CHANGE_AVAILABLE:
        if (last_tile_number == changed_region.tile_number) {
          return result;
        }
        tile_data.free_data();
        tile_data.init_data(changed_region.tile_number);
        last_tile_number = changed_region.tile_number;
        return result;

        ;
      default:
        BLI_assert_unreachable();
        return result;
    }
  }
};

template<typename TileData = NoTileData> class PartialUpdateChecker {

  Image *image;
  ImageUser *image_user;
  PartialUpdateUser *user;

 public:
  PartialUpdateChecker(Image *image, ImageUser *image_user, PartialUpdateUser *user)
      : image(image), image_user(image_user), user(user)
  {
  }

  PartialUpdateCollectResult<TileData> collect_changes()
  {
    ePartialUpdateCollectResult collect_result = BKE_image_partial_update_collect_changes(image,
                                                                                          user);
    return PartialUpdateCollectResult<TileData>(image, *image_user, user, collect_result);
  }
};

}  // namespace partial_update
}  // namespace blender::bke::image