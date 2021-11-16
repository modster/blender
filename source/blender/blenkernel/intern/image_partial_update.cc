
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

#include "BKE_image.h"

#include "DNA_image_types.h"

#include "IMB_imbuf.h"
#include "IMB_imbuf_types.h"

#include "BLI_vector.hh"

namespace blender::bke::image::partial_update {

struct PartialUpdateUserImpl;
struct PartialUpdateRegisterImpl;

/**
 * Wrap the CPP implementation (PartialUpdateUserImpl) to a C struct.
 */
static struct PartialUpdateUser *wrap(PartialUpdateUserImpl *user)
{
  return static_cast<struct PartialUpdateUser *>(static_cast<void *>(user));
}
static PartialUpdateUserImpl *unwrap(struct PartialUpdateUser *user)
{
  return static_cast<PartialUpdateUserImpl *>(static_cast<void *>(user));
}
static struct PartialUpdateRegister *wrap(PartialUpdateRegisterImpl *partial_update_register)
{
  return static_cast<struct PartialUpdateRegister *>(static_cast<void *>(partial_update_register));
}
static PartialUpdateRegisterImpl *unwrap(struct PartialUpdateRegister *partial_update_register)
{
  return static_cast<PartialUpdateRegisterImpl *>(static_cast<void *>(partial_update_register));
}

using TransactionID = int64_t;
constexpr TransactionID UnknownTransactionID = -1;

struct PartialUpdateUserImpl {
  /** \brief last transaction id that was seen by this user. */
  TransactionID last_transaction_id = UnknownTransactionID;

  /** \brief tiles that have been updated. */
  Vector<PartialUpdateTile> updated_tiles;

  /**
   * \brief Clear the updated tiles.
   *
   * Updated tiles should be cleared at the start of #BKE_image_partial_update_collect_tiles so the
   */
  void clear_updated_tiles()
  {
    updated_tiles.clear();
  }
};

struct PartialUpdateTransaction {
  /* bitvec for each tile in the transaction. True means that this tile was changed during during
   * this transaction. */
  Vector<bool> tile_validity;
  int tile_x_len_;
  int tile_y_len_;
  bool is_empty_ = true;

  bool is_empty()
  {
    return is_empty_;
  }

  void init_tiles(int tile_x_len, int tile_y_len)
  {
    tile_x_len_ = tile_x_len;
    tile_y_len_ = tile_y_len;
    const int tile_len = tile_x_len * tile_y_len;
    tile_validity.resize(tile_len);
    /* Fast exit. When the transaction was already empty no need to re-init the tile_validity. */
    if (is_empty()) {
      return;
    }
    for (int index = 0; index < tile_len; index++) {
      tile_validity[index] = false;
    }
    is_empty_ = true;
  }

  void reset()
  {
    init_tiles(tile_x_len_, tile_y_len_);
  }

  void add_region(int start_x_tile, int start_y_tile, int end_x_tile, int end_y_tile)
  {
    for (int tile_y = start_y_tile; tile_y <= end_y_tile; tile_y++) {
      for (int tile_x = start_x_tile; tile_x <= end_x_tile; tile_x++) {
        int tile_index = tile_y * tile_x_len_ + tile_x;
        tile_validity[tile_index] = true;
      }
    }
    is_empty_ = false;
  }

  /** \brief Merge the given transaction into the receiver. */
  void merge(PartialUpdateTransaction &other)
  {
    BLI_assert(tile_x_len_ == other.tile_x_len_);
    BLI_assert(tile_y_len_ == other.tile_y_len_);
    const int tile_len = tile_x_len_ * tile_y_len_;

    for (int tile_index = 0; tile_index < tile_len; tile_index++) {
      tile_validity[tile_index] |= other.tile_validity[tile_index];
    }
    if (!other.is_empty()) {
      is_empty_ = false;
    }
  }

  /** \brief has a tile changed inside this transaction. */
  bool tile_changed(int tile_x, int tile_y)
  {
    int tile_index = tile_y * tile_x_len_ + tile_x;
    return tile_validity[tile_index];
  }
};

/** \brief Partial update data that is stored inside the image. */
struct PartialUpdateRegisterImpl {
  /* Changes are tracked in tiles. */
  static constexpr int TILE_SIZE = 256;

  TransactionID first_transaction_id;
  TransactionID current_transaction_id;
  Vector<PartialUpdateTransaction> history;

  PartialUpdateTransaction current_transaction;

  int image_width;
  int image_height;

  void set_resolution(ImBuf *image_buffer)
  {
    if (image_width != image_buffer->x || image_height != image_buffer->y) {
      image_width = image_buffer->x;
      image_height = image_buffer->y;

      int tile_x_len = image_width / TILE_SIZE;
      int tile_y_len = image_height / TILE_SIZE;
      current_transaction.init_tiles(tile_x_len, tile_y_len);

      mark_full_update();
    }
  }

  void mark_full_update()
  {
    history.clear();
    current_transaction_id++;
    current_transaction.reset();
    first_transaction_id = current_transaction_id;
  }

  void mark_region(rcti *updated_region)
  {
    int start_x_tile = updated_region->xmin / TILE_SIZE;
    int end_x_tile = updated_region->xmax / TILE_SIZE;
    int start_y_tile = updated_region->ymin / TILE_SIZE;
    int end_y_tile = updated_region->ymax / TILE_SIZE;

    /* Clamp tiles to tiles in image. */
    start_x_tile = max_ii(0, start_x_tile);
    start_y_tile = max_ii(0, start_y_tile);
    end_x_tile = min_ii(current_transaction.tile_x_len_ - 1, end_x_tile);
    end_y_tile = min_ii(current_transaction.tile_y_len_ - 1, end_y_tile);

    /* Early exit when no tiles need to be updated. */
    if (start_x_tile >= current_transaction.tile_x_len_) {
      return;
    }
    if (start_y_tile >= current_transaction.tile_y_len_) {
      return;
    }
    if (end_x_tile < 0) {
      return;
    }
    if (end_y_tile < 0) {
      return;
    }

    current_transaction.add_region(start_x_tile, start_y_tile, end_x_tile, end_y_tile);
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
    current_transaction.reset();
    current_transaction_id++;
  }

  /**
   * /brief Check if data is available to construct the update tiles for the given
   * transaction_id.
   *
   * The update tiles can be created when transaction id is between
   */
  bool can_construct(TransactionID transaction_id)
  {
    return transaction_id >= first_transaction_id;
  }

  std::unique_ptr<PartialUpdateTransaction> changed_tiles_since(
      const TransactionID from_transaction)
  {
    std::unique_ptr<PartialUpdateTransaction> changed_tiles =
        std::make_unique<PartialUpdateTransaction>();
    int tile_x_len = image_width / TILE_SIZE;
    int tile_y_len = image_height / TILE_SIZE;
    changed_tiles->init_tiles(tile_x_len, tile_y_len);

    for (int index = from_transaction - first_transaction_id; index < history.size(); index++) {
      changed_tiles->merge(history[index]);
    }
    return changed_tiles;
  }
};

}  // namespace blender::bke::image::partial_update

extern "C" {

using namespace blender::bke::image::partial_update;

struct PartialUpdateUser *BKE_image_partial_update_create(struct Image *image)
{
  BLI_assert(image);
  PartialUpdateUserImpl *user_impl = OBJECT_GUARDED_NEW(PartialUpdateUserImpl);
  return wrap(user_impl);
}

void BKE_image_partial_update_free(PartialUpdateUser *user)
{
  PartialUpdateUserImpl *user_impl = unwrap(user);
  OBJECT_GUARDED_DELETE(user_impl, PartialUpdateUserImpl);
}

ePartialUpdateCollectResult BKE_image_partial_update_collect_tiles(Image *image,
                                                                   ImBuf *image_buffer,
                                                                   PartialUpdateUser *user)
{
  PartialUpdateUserImpl *user_impl = unwrap(user);
  user_impl->clear_updated_tiles();

  PartialUpdateRegisterImpl *partial_updater = unwrap(
      BKE_image_partial_update_register_ensure(image, image_buffer));
  partial_updater->ensure_empty_transaction();

  if (!partial_updater->can_construct(user_impl->last_transaction_id)) {
    user_impl->last_transaction_id = partial_updater->current_transaction_id;
    return PARTIAL_UPDATE_NEED_FULL_UPDATE;
  }

  if (user_impl->last_transaction_id == partial_updater->current_transaction_id) {
    // No changes since last time.
    return PARTIAL_UPDATE_NO_CHANGES;
  }

  // TODO: Collect changes between last_transaction_id and current_transaction_id.
  std::unique_ptr<PartialUpdateTransaction> changed_tiles = partial_updater->changed_tiles_since(
      user_impl->last_transaction_id);
  for (int tile_y = 0; tile_y < changed_tiles->tile_y_len_; tile_y++) {
    for (int tile_x = 0; tile_x < changed_tiles->tile_x_len_; tile_x++) {
      if (changed_tiles->tile_changed(tile_x, tile_y)) {
        PartialUpdateTile tile;
        BLI_rcti_init(&tile.region,
                      tile_x * PartialUpdateRegisterImpl::TILE_SIZE,
                      (tile_x + 1) * PartialUpdateRegisterImpl::TILE_SIZE,
                      tile_y * PartialUpdateRegisterImpl::TILE_SIZE,
                      (tile_y + 1) * PartialUpdateRegisterImpl::TILE_SIZE);
        user_impl->updated_tiles.append_as(tile);
      }
    }
  }

  // TODO: compress neighboring tiles and store in user.

  user_impl->last_transaction_id = partial_updater->current_transaction_id;
  return PARTIAL_UPDATE_CHANGES_AVAILABLE;
}

ePartialUpdateIterResult BKE_image_partial_update_next_tile(PartialUpdateUser *user,
                                                            PartialUpdateTile *r_tile)
{
  PartialUpdateUserImpl *user_impl = unwrap(user);
  if (user_impl->updated_tiles.is_empty()) {
    return PARTIAL_UPDATE_ITER_NO_TILES_LEFT;
  }
  PartialUpdateTile tile = user_impl->updated_tiles.pop_last();
  *r_tile = tile;
  return PARTIAL_UPDATE_ITER_TILE_LOADED;
}

/* --- Image side --- */

struct PartialUpdateRegister *BKE_image_partial_update_register_ensure(Image *image,
                                                                       ImBuf *image_buffer)
{
  if (image->runtime.partial_update_register == nullptr) {
    PartialUpdateRegisterImpl *partial_update_register = OBJECT_GUARDED_NEW(
        PartialUpdateRegisterImpl);
    partial_update_register->set_resolution(image_buffer);
    image->runtime.partial_update_register = wrap(partial_update_register);
  }
  return image->runtime.partial_update_register;
}

void BKE_image_partial_update_register_free(Image *image)
{
  PartialUpdateRegisterImpl *partial_update_register = unwrap(
      image->runtime.partial_update_register);
  if (partial_update_register) {
    OBJECT_GUARDED_DELETE(partial_update_register, PartialUpdateRegisterImpl);
  }
  image->runtime.partial_update_register = nullptr;
}

void BKE_image_partial_update_register_mark_region(Image *image,
                                                   ImBuf *image_buffer,
                                                   rcti *updated_region)
{
  PartialUpdateRegisterImpl *partial_updater = unwrap(
      BKE_image_partial_update_register_ensure(image, image_buffer));
  partial_updater->mark_region(updated_region);
}

void BKE_image_partial_update_register_mark_full_update(Image *image, ImBuf *image_buffer)
{
  PartialUpdateRegisterImpl *partial_updater = unwrap(
      BKE_image_partial_update_register_ensure(image, image_buffer));
  partial_updater->mark_full_update();
  partial_updater->set_resolution(image_buffer);
}
}
