
/**
 * \file image_gpu_partial_update.cc
 *
 * To reduce the overhead of image processing this file contains a mechanism to detect areas of the
 * image that are changed. These areas are organized in tiles. Changes that happen over time are
 * organized in changesets.
 *
 * A common usecase is to update GPUTexture for drawing where only that part is uploaded that only
 * changed.
 *
 * Usage:
 *
 * ```
 * Image *image = ...;
 * ImBuf *image_buffer = ...;
 *
 * // partial_update_user should be kept for the whole session where the changes needs to be
 * // tracked. Keep this instance alive as long as you need to track image changes.
 *
 * PartialUpdateUser *partial_update_user = BKE_image_partial_update_create(image);
 *
 * ...
 *
 * switch (BKE_image_partial_update_collect_tiles(image, image_buffer))
 * {
 * case PARTIAL_UPDATE_NEED_FULL_UPDATE:
 *  // Unable to do partial updates. Perform a full update.
 *  break;
 * case PARTIAL_UPDATE_CHANGES_AVAILABLE:
 *  PartialUpdateTile tile;
 *  while (BKE_image_partial_update_next_tile(partial_update_user, &tile) ==
 *         PARTIAL_UPDATE_ITER_TILE_LOADED){
 *  // Do something with the tile.
 *  }
 *  case PARTIAL_UPDATE_NO_CHANGES:
 *    break;
 * }
 *
 * ...
 *
 * // Free partial_update_user.
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
 * Wrap PartialUpdateUserImpl to its C-struct (PartialUpdateUser).
 */
static struct PartialUpdateUser *wrap(PartialUpdateUserImpl *user)
{
  return static_cast<struct PartialUpdateUser *>(static_cast<void *>(user));
}

/**
 * Unwrap the PartialUpdateUser C-struct to its CPP counterpart (PartialUpdateUserImpl).
 */
static PartialUpdateUserImpl *unwrap(struct PartialUpdateUser *user)
{
  return static_cast<PartialUpdateUserImpl *>(static_cast<void *>(user));
}

/**
 * Wrap PartialUpdateRegisterImpl to its C-struct (PartialUpdateRegister).
 */
static struct PartialUpdateRegister *wrap(PartialUpdateRegisterImpl *partial_update_register)
{
  return static_cast<struct PartialUpdateRegister *>(static_cast<void *>(partial_update_register));
}

/**
 * Unwrap the PartialUpdateRegister C-struct to its CPP counterpart (PartialUpdateRegisterImpl).
 */
static PartialUpdateRegisterImpl *unwrap(struct PartialUpdateRegister *partial_update_register)
{
  return static_cast<PartialUpdateRegisterImpl *>(static_cast<void *>(partial_update_register));
}

using ChangesetID = int64_t;
constexpr ChangesetID UnknownChangesetID = -1;

struct PartialUpdateUserImpl {
  /** \brief last changeset id that was seen by this user. */
  ChangesetID last_changeset_id = UnknownChangesetID;

  /** \brief tiles that have been updated. */
  Vector<PartialUpdateTile> updated_tiles;

  /**
   * \brief Clear the updated tiles.
   *
   * Updated tiles should be cleared at the start of #BKE_image_partial_update_collect_tiles so
   * the
   */
  void clear_updated_tiles()
  {
    updated_tiles.clear();
  }
};

/**
 * \brief Dirty tiles.
 *
 * Internally dirty tiles are grouped together in change sets to make sure that the correct
 * answer can be built for different users reducing the amount of merges.
 */
struct TileChangeset {
 private:
  /** \brief Dirty flag for each tile. */
  std::vector<bool> tile_dirty_flags_;
  /** \brief are there dirty/ */
  bool has_dirty_tiles_ = false;

 public:
  /** \brief Number of tiles along the x-axis. */
  int tile_x_len_;
  /** \brief Number of tiles along the y-axis. */
  int tile_y_len_;

  bool has_dirty_tiles() const
  {
    return has_dirty_tiles_;
  }

  void init_tiles(int tile_x_len, int tile_y_len)
  {
    tile_x_len_ = tile_x_len;
    tile_y_len_ = tile_y_len;
    const int tile_len = tile_x_len * tile_y_len;
    const int previous_tile_len = tile_dirty_flags_.size();

    tile_dirty_flags_.resize(tile_len);
    /* Fast exit. When the changeset was already empty no need to re-init the tile_validity. */
    if (!has_dirty_tiles()) {
      return;
    }
    for (int index = 0; index < min_ii(tile_len, previous_tile_len); index++) {
      tile_dirty_flags_[index] = false;
    }
    has_dirty_tiles_ = false;
  }

  void reset()
  {
    init_tiles(tile_x_len_, tile_y_len_);
  }

  void mark_tiles_dirty(int start_x_tile, int start_y_tile, int end_x_tile, int end_y_tile)
  {
    for (int tile_y = start_y_tile; tile_y <= end_y_tile; tile_y++) {
      for (int tile_x = start_x_tile; tile_x <= end_x_tile; tile_x++) {
        int tile_index = tile_y * tile_x_len_ + tile_x;
        tile_dirty_flags_[tile_index] = true;
      }
    }
    has_dirty_tiles_ = true;
  }

  /** \brief Merge the given changeset into the receiver. */
  void merge(const TileChangeset &other)
  {
    BLI_assert(tile_x_len_ == other.tile_x_len_);
    BLI_assert(tile_y_len_ == other.tile_y_len_);
    const int tile_len = tile_x_len_ * tile_y_len_;

    for (int tile_index = 0; tile_index < tile_len; tile_index++) {
      tile_dirty_flags_[tile_index] = tile_dirty_flags_[tile_index] |
                                      other.tile_dirty_flags_[tile_index];
    }
    has_dirty_tiles_ |= other.has_dirty_tiles_;
  }

  /** \brief has a tile changed inside this changeset. */
  bool is_tile_dirty(int tile_x, int tile_y) const
  {
    const int tile_index = tile_y * tile_x_len_ + tile_x;
    return tile_dirty_flags_[tile_index];
  }
};

/**
 * \brief Partial update changes stored inside the image runtime.
 *
 * The PartialUpdateRegisterImpl will keep track of changes over time. Changes are groups inside
 * TileChangesets.
 */
struct PartialUpdateRegisterImpl {
  /* Changes are tracked in tiles. */
  static constexpr int TILE_SIZE = 256;

  /** \brief changeset id of the first changeset kept in #history. */
  ChangesetID first_changeset_id;
  /** \brief changeset id of the top changeset kept in #history. */
  ChangesetID last_changeset_id;

  /** \brief history of changesets. */
  Vector<TileChangeset> history;
  /** \brief The current changeset. New changes will be added to this changeset only. */
  TileChangeset current_changeset;

  int image_width;
  int image_height;

  void set_resolution(ImBuf *image_buffer)
  {
    if (image_width != image_buffer->x || image_height != image_buffer->y) {
      image_width = image_buffer->x;
      image_height = image_buffer->y;

      int tile_x_len = image_width / TILE_SIZE;
      int tile_y_len = image_height / TILE_SIZE;
      current_changeset.init_tiles(tile_x_len, tile_y_len);

      mark_full_update();
    }
  }

  void mark_full_update()
  {
    history.clear();
    last_changeset_id++;
    current_changeset.reset();
    first_changeset_id = last_changeset_id;
  }

  /**
   * \brief get the tile number for the give pixel coordinate.
   *
   * As tiles are squares the this member can be used for both x and y axis.
   */
  static int tile_number_for_pixel(int pixel_offset)
  {
    int tile_offset = pixel_offset / TILE_SIZE;
    if (pixel_offset < 0) {
      tile_offset -= 1;
    }
    return tile_offset;
  }

  void mark_region(rcti *updated_region)
  {
    int start_x_tile = tile_number_for_pixel(updated_region->xmin);
    int end_x_tile = tile_number_for_pixel(updated_region->xmax - 1);
    int start_y_tile = tile_number_for_pixel(updated_region->ymin);
    int end_y_tile = tile_number_for_pixel(updated_region->ymax - 1);

    /* Clamp tiles to tiles in image. */
    start_x_tile = max_ii(0, start_x_tile);
    start_y_tile = max_ii(0, start_y_tile);
    end_x_tile = min_ii(current_changeset.tile_x_len_ - 1, end_x_tile);
    end_y_tile = min_ii(current_changeset.tile_y_len_ - 1, end_y_tile);

    /* Early exit when no tiles need to be updated. */
    if (start_x_tile >= current_changeset.tile_x_len_) {
      return;
    }
    if (start_y_tile >= current_changeset.tile_y_len_) {
      return;
    }
    if (end_x_tile < 0) {
      return;
    }
    if (end_y_tile < 0) {
      return;
    }

    current_changeset.mark_tiles_dirty(start_x_tile, start_y_tile, end_x_tile, end_y_tile);
  }

  void ensure_empty_changeset()
  {
    if (!current_changeset.has_dirty_tiles()) {
      /* No need to create a new changeset when previous changeset does not contain any dirty
       * tiles. */
      return;
    }
    commit_current_changeset();
  }

  /** Move the current changeset to the history and resets the current changeset. */
  void commit_current_changeset()
  {
    history.append_as(std::move(current_changeset));
    current_changeset.reset();
    last_changeset_id++;
  }

  /**
   * /brief Check if data is available to construct the update tiles for the given
   * changeset_id.
   *
   * The update tiles can be created when changeset id is between
   */
  bool can_construct(ChangesetID changeset_id)
  {
    return changeset_id >= first_changeset_id;
  }

  /**
   * \brief collect all historic changes since a given changeset.
   */
  std::unique_ptr<TileChangeset> changed_tiles_since(const ChangesetID from_changeset)
  {
    std::unique_ptr<TileChangeset> changed_tiles = std::make_unique<TileChangeset>();
    int tile_x_len = image_width / TILE_SIZE;
    int tile_y_len = image_height / TILE_SIZE;
    changed_tiles->init_tiles(tile_x_len, tile_y_len);

    for (int index = from_changeset - first_changeset_id; index < history.size(); index++) {
      changed_tiles->merge(history[index]);
    }
    return changed_tiles;
  }
};

}  // namespace blender::bke::image::partial_update

extern "C" {

using namespace blender::bke::image::partial_update;

// TODO(jbakker): cleanup parameter.
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
  partial_updater->ensure_empty_changeset();

  if (!partial_updater->can_construct(user_impl->last_changeset_id)) {
    user_impl->last_changeset_id = partial_updater->last_changeset_id;
    return PARTIAL_UPDATE_NEED_FULL_UPDATE;
  }

  /* Check if there are changes since last invocation for the user. */
  if (user_impl->last_changeset_id == partial_updater->last_changeset_id) {
    return PARTIAL_UPDATE_NO_CHANGES;
  }

  /* Collect changed tiles. */
  std::unique_ptr<TileChangeset> changed_tiles = partial_updater->changed_tiles_since(
      user_impl->last_changeset_id);

  /* Convert tiles in the changeset to rectangles that are dirty. */
  for (int tile_y = 0; tile_y < changed_tiles->tile_y_len_; tile_y++) {
    for (int tile_x = 0; tile_x < changed_tiles->tile_x_len_; tile_x++) {
      if (!changed_tiles->is_tile_dirty(tile_x, tile_y)) {
        continue;
      }

      PartialUpdateTile tile;
      BLI_rcti_init(&tile.region,
                    tile_x * PartialUpdateRegisterImpl::TILE_SIZE,
                    (tile_x + 1) * PartialUpdateRegisterImpl::TILE_SIZE,
                    tile_y * PartialUpdateRegisterImpl::TILE_SIZE,
                    (tile_y + 1) * PartialUpdateRegisterImpl::TILE_SIZE);
      user_impl->updated_tiles.append_as(tile);
    }
  }

  user_impl->last_changeset_id = partial_updater->last_changeset_id;
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
