
/**
 * \file image_gpu_partial_update.cc
 *
 * To reduce the overhead of uploading images to GPU only changed areas will be uploaded.
 * The areas are organized in tiles.
 *
 * Requirements:
 * - Independent how the actual GPU textures look like. The uploading, transforming are
 *   responsibility of the user of this API.
 *
 *
 * Usage:
 *
 * ```
 * Image *image = space_data->image;
 * PartialUpdateUser* partial_update_user = BKE_image_partial_update_create(image);
 *
 * if (BKE_image_partial_update_collect_tiles(image, partial_update_user) == DO_FULL_UPDATE {
 *   // Recreate full GPU texture.
 *   ...
 * } else {
 *   PartialUpdateTile tile;
 *  int index = 0;
 *   while(BKE_image_partial_update_get_tile(partial_update_user, index++, &tile) == TILE_VALID) {
 *     // Do something with the tile.
 *     ...
 *   }
 * }
 *
 * BKE_image_partial_update_free(partial_update_user);
 *
 * ```
 */

#include "DNA_image_types.h"

#include "BLI_vector.hh"

namespace blender::bke::image {

/**
 * \brief A single tile to update.
 *
 * Data is organized in tiles. These tiles are in texel space (1 unit is a single texel). When
 * tiles are requested they are merged with neighboring tiles.
 */
struct PartialUpdateTile {
  /** \brief area of the image that has been updated. */
  rcti update_area;
};

using TransactionID = int;

struct PartialUpdateUser {
 private:
  TransactionID last_transaction_id;

  /** \brief tiles that have been updated. */
  Vector<PartialUpdateTile> changed_tiles;
};

struct PartialUpdateTransaction {
  /* bitvec for each tile in the transaction. True means that this tile was changed during during
   * this transaction. */
  Vector<bool> tile_validity;
  bool is_empty = true;

  bool is_empty()
  {
    return is_empty;
  }
};

/** \brief Partial update data that is stored inside the image. */
struct PartialUpdateImage {
  TransactionID first_transaction_id;
  TransactionID current_transaction_id;
  Vector<PartialUpdateTransaction> history;

  PartialUpdateTransaction current_transaction;

  void invalidate_tile(int tile_x, int tile_y)
  {
  }

  void ensure_empty_transaction()
  {
    if (current_transaction.is_empty()) {
      /* No need to create a new transaction when previous transaction does not contain any data.
       */
      return;
    }
    commit_current_transaction();
  }

  void commit_current_transaction()
  {
    history.append_as(std::move(current_transaction));
    current_transaction_id++;
  }
}

}  // namespace blender::bke::image

extern "C" {
struct PartialUpdateUser;
using namespace blender::bke::image;

PartialUpdateUser *BKE_image_partial_update_create(Image *image)
{
  return nullptr;
}

void BKE_image_partial_update_free(PartialUpdateUser *user)
{
}

bool BKE_image_partial_update_collect_tiles(Image *image, PartialUpdateUser *user)
{
  user->clear_updated_tiles();
  PartialUpdateImage *partial_updater = nullptr;  //*image->partial_updater;
  partial_updater->ensure_empty_transaction();

  if (user->last_transaction_id < partial_updater->first_transaction_id) {
    user->last_transaction_id = partial_updater->current_transaction_id;
    return NEED_FULL_UPDATE;
  }

  if (user->last_transaction_id == partial_updater->current_transaction_id) {
    // No changes since last time.
    return NO_CHANGES_SINCE_LAST_TIME;
  }

  // TODO: Collect changes between last_transaction_id and current_transaction_id.
  // TODO: compress neighboring tiles and store in user.

  user->last_transaction_id = partial_updated->current_transaction_id;
  return CHANGES_DETECTED;
}

bool BKE_image_partial_update_get_tile(PartialUpdateUser *user,
                                       int index,
                                       PartialUpdateTile *r_tile)
{
  return NO_TILES_LEFT;
}

}