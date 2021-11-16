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
 * The Original Code is Copyright (C) 2020 by Blender Foundation.
 */
#include "testing/testing.h"

#include "CLG_log.h"

#include "BKE_appdir.h"
#include "BKE_idtype.h"
#include "BKE_image.h"
#include "BKE_main.h"

#include "IMB_imbuf.h"

#include "DNA_image_types.h"

#include "MEM_guardedalloc.h"

namespace blender::bke::image {

constexpr float black_color[4] = {0.0f, 0.0f, 0.0f, 1.0f};

class ImagePartialUpdateTest : public testing::Test {
 protected:
  Main *bmain;
  Image *image;
  ImBuf *image_buffer;
  PartialUpdateUser *partial_update_user;

 private:
  Image *create_test_image(int width, int height)
  {
    return BKE_image_add_generated(bmain,
                                   width,
                                   height,
                                   "Test Image",
                                   32,
                                   true,
                                   IMA_GENTYPE_BLANK,
                                   black_color,
                                   false,
                                   false,
                                   false);
  }

 protected:
  void SetUp() override
  {
    CLG_init();
    BKE_idtype_init();
    BKE_appdir_init();
    IMB_init();

    bmain = BKE_main_new();
    /* Creating an image generates a mem-leak during tests. */
    image = create_test_image(1024, 1024);
    image_buffer = BKE_image_acquire_ibuf(image, nullptr, nullptr);

    partial_update_user = BKE_image_partial_update_create(image);
  }

  void TearDown() override
  {
    BKE_image_release_ibuf(image, image_buffer, nullptr);
    BKE_image_partial_update_free(partial_update_user);
    BKE_main_free(bmain);

    IMB_exit();
    BKE_appdir_exit();
    CLG_exit();
  }
};

TEST_F(ImagePartialUpdateTest, mark_full_update)
{
  ePartialUpdateCollectResult result;
  /* First tile should always return a full update. */
  result = BKE_image_partial_update_collect_tiles(image, image_buffer, partial_update_user);
  EXPECT_EQ(result, PARTIAL_UPDATE_NEED_FULL_UPDATE);
  /* Second invoke should now detect no changes. */
  result = BKE_image_partial_update_collect_tiles(image, image_buffer, partial_update_user);
  EXPECT_EQ(result, PARTIAL_UPDATE_NO_CHANGES);

  /* Mark full update */
  BKE_image_partial_update_register_mark_full_update(image, image_buffer);

  /* Validate need full update followed by no changes. */
  result = BKE_image_partial_update_collect_tiles(image, image_buffer, partial_update_user);
  EXPECT_EQ(result, PARTIAL_UPDATE_NEED_FULL_UPDATE);
  result = BKE_image_partial_update_collect_tiles(image, image_buffer, partial_update_user);
  EXPECT_EQ(result, PARTIAL_UPDATE_NO_CHANGES);
}

TEST_F(ImagePartialUpdateTest, mark_single_tile)
{
  ePartialUpdateCollectResult result;
  /* First tile should always return a full update. */
  result = BKE_image_partial_update_collect_tiles(image, image_buffer, partial_update_user);
  EXPECT_EQ(result, PARTIAL_UPDATE_NEED_FULL_UPDATE);
  /* Second invoke should now detect no changes. */
  result = BKE_image_partial_update_collect_tiles(image, image_buffer, partial_update_user);
  EXPECT_EQ(result, PARTIAL_UPDATE_NO_CHANGES);

  /* Mark region. */
  rcti region;
  BLI_rcti_init(&region, 10, 20, 40, 50);
  BKE_image_partial_update_register_mark_region(image, image_buffer, &region);

  /* Partial Update should be available. */
  result = BKE_image_partial_update_collect_tiles(image, image_buffer, partial_update_user);
  EXPECT_EQ(result, PARTIAL_UPDATE_CHANGES_AVAILABLE);

  /* Check tiles. */
  PartialUpdateTile changed_tile;
  ePartialUpdateIterResult iter_result;
  iter_result = BKE_image_partial_update_next_tile(partial_update_user, &changed_tile);
  EXPECT_EQ(iter_result, PARTIAL_UPDATE_ITER_TILE_LOADED);
  EXPECT_EQ(BLI_rcti_inside_rcti(&changed_tile.region, &region), true);
  iter_result = BKE_image_partial_update_next_tile(partial_update_user, &changed_tile);
  EXPECT_EQ(iter_result, PARTIAL_UPDATE_ITER_NO_TILES_LEFT);

  result = BKE_image_partial_update_collect_tiles(image, image_buffer, partial_update_user);
  EXPECT_EQ(result, PARTIAL_UPDATE_NO_CHANGES);
}

TEST_F(ImagePartialUpdateTest, mark_unconnected_tiles)
{
  ePartialUpdateCollectResult result;
  /* First tile should always return a full update. */
  result = BKE_image_partial_update_collect_tiles(image, image_buffer, partial_update_user);
  EXPECT_EQ(result, PARTIAL_UPDATE_NEED_FULL_UPDATE);
  /* Second invoke should now detect no changes. */
  result = BKE_image_partial_update_collect_tiles(image, image_buffer, partial_update_user);
  EXPECT_EQ(result, PARTIAL_UPDATE_NO_CHANGES);

  /* Mark region. */
  rcti region_a;
  BLI_rcti_init(&region_a, 10, 20, 40, 50);
  BKE_image_partial_update_register_mark_region(image, image_buffer, &region_a);
  rcti region_b;
  BLI_rcti_init(&region_b, 710, 720, 740, 750);
  BKE_image_partial_update_register_mark_region(image, image_buffer, &region_b);

  /* Partial Update should be available. */
  result = BKE_image_partial_update_collect_tiles(image, image_buffer, partial_update_user);
  EXPECT_EQ(result, PARTIAL_UPDATE_CHANGES_AVAILABLE);

  /* Check tiles. */
  PartialUpdateTile changed_tile;
  ePartialUpdateIterResult iter_result;
  iter_result = BKE_image_partial_update_next_tile(partial_update_user, &changed_tile);
  EXPECT_EQ(iter_result, PARTIAL_UPDATE_ITER_TILE_LOADED);
  EXPECT_EQ(BLI_rcti_inside_rcti(&changed_tile.region, &region_b), true);
  iter_result = BKE_image_partial_update_next_tile(partial_update_user, &changed_tile);
  EXPECT_EQ(iter_result, PARTIAL_UPDATE_ITER_TILE_LOADED);
  EXPECT_EQ(BLI_rcti_inside_rcti(&changed_tile.region, &region_a), true);
  iter_result = BKE_image_partial_update_next_tile(partial_update_user, &changed_tile);
  EXPECT_EQ(iter_result, PARTIAL_UPDATE_ITER_NO_TILES_LEFT);

  result = BKE_image_partial_update_collect_tiles(image, image_buffer, partial_update_user);
  EXPECT_EQ(result, PARTIAL_UPDATE_NO_CHANGES);
}

TEST_F(ImagePartialUpdateTest, donot_mark_outside_image)
{
  ePartialUpdateCollectResult result;
  /* First tile should always return a full update. */
  result = BKE_image_partial_update_collect_tiles(image, image_buffer, partial_update_user);
  EXPECT_EQ(result, PARTIAL_UPDATE_NEED_FULL_UPDATE);
  /* Second invoke should now detect no changes. */
  result = BKE_image_partial_update_collect_tiles(image, image_buffer, partial_update_user);
  EXPECT_EQ(result, PARTIAL_UPDATE_NO_CHANGES);

  /* Mark region. */
  rcti region;
  /* Axis. */
  BLI_rcti_init(&region, -100, 0, 50, 100);
  BKE_image_partial_update_register_mark_region(image, image_buffer, &region);
  result = BKE_image_partial_update_collect_tiles(image, image_buffer, partial_update_user);
  EXPECT_EQ(result, PARTIAL_UPDATE_NO_CHANGES);

  BLI_rcti_init(&region, 1024, 1100, 50, 100);
  BKE_image_partial_update_register_mark_region(image, image_buffer, &region);
  result = BKE_image_partial_update_collect_tiles(image, image_buffer, partial_update_user);
  EXPECT_EQ(result, PARTIAL_UPDATE_NO_CHANGES);

  BLI_rcti_init(&region, 50, 100, -100, 0);
  BKE_image_partial_update_register_mark_region(image, image_buffer, &region);
  result = BKE_image_partial_update_collect_tiles(image, image_buffer, partial_update_user);
  EXPECT_EQ(result, PARTIAL_UPDATE_NO_CHANGES);

  BLI_rcti_init(&region, 50, 100, 1024, 1100);
  BKE_image_partial_update_register_mark_region(image, image_buffer, &region);
  result = BKE_image_partial_update_collect_tiles(image, image_buffer, partial_update_user);
  EXPECT_EQ(result, PARTIAL_UPDATE_NO_CHANGES);

  /* Diagonals. */
  BLI_rcti_init(&region, -100, 0, -100, 0);
  BKE_image_partial_update_register_mark_region(image, image_buffer, &region);
  result = BKE_image_partial_update_collect_tiles(image, image_buffer, partial_update_user);
  EXPECT_EQ(result, PARTIAL_UPDATE_NO_CHANGES);

  BLI_rcti_init(&region, -100, 0, 1024, 1100);
  BKE_image_partial_update_register_mark_region(image, image_buffer, &region);
  result = BKE_image_partial_update_collect_tiles(image, image_buffer, partial_update_user);
  EXPECT_EQ(result, PARTIAL_UPDATE_NO_CHANGES);

  BLI_rcti_init(&region, 1024, 1100, -100, 0);
  BKE_image_partial_update_register_mark_region(image, image_buffer, &region);
  result = BKE_image_partial_update_collect_tiles(image, image_buffer, partial_update_user);
  EXPECT_EQ(result, PARTIAL_UPDATE_NO_CHANGES);

  BLI_rcti_init(&region, 1024, 1100, 1024, 1100);
  BKE_image_partial_update_register_mark_region(image, image_buffer, &region);
  result = BKE_image_partial_update_collect_tiles(image, image_buffer, partial_update_user);
  EXPECT_EQ(result, PARTIAL_UPDATE_NO_CHANGES);
}

TEST_F(ImagePartialUpdateTest, mark_inside_image)
{
  ePartialUpdateCollectResult result;
  /* First tile should always return a full update. */
  result = BKE_image_partial_update_collect_tiles(image, image_buffer, partial_update_user);
  EXPECT_EQ(result, PARTIAL_UPDATE_NEED_FULL_UPDATE);
  /* Second invoke should now detect no changes. */
  result = BKE_image_partial_update_collect_tiles(image, image_buffer, partial_update_user);
  EXPECT_EQ(result, PARTIAL_UPDATE_NO_CHANGES);

  /* Mark region. */
  rcti region;
  BLI_rcti_init(&region, 0, 1, 0, 1);
  BKE_image_partial_update_register_mark_region(image, image_buffer, &region);
  result = BKE_image_partial_update_collect_tiles(image, image_buffer, partial_update_user);
  EXPECT_EQ(result, PARTIAL_UPDATE_CHANGES_AVAILABLE);

  result = BKE_image_partial_update_collect_tiles(image, image_buffer, partial_update_user);
  EXPECT_EQ(result, PARTIAL_UPDATE_NO_CHANGES);
  BLI_rcti_init(&region, 1023, 1024, 0, 1);
  BKE_image_partial_update_register_mark_region(image, image_buffer, &region);
  result = BKE_image_partial_update_collect_tiles(image, image_buffer, partial_update_user);
  EXPECT_EQ(result, PARTIAL_UPDATE_CHANGES_AVAILABLE);

  result = BKE_image_partial_update_collect_tiles(image, image_buffer, partial_update_user);
  EXPECT_EQ(result, PARTIAL_UPDATE_NO_CHANGES);
  BLI_rcti_init(&region, 1023, 1024, 1023, 1024);
  BKE_image_partial_update_register_mark_region(image, image_buffer, &region);
  result = BKE_image_partial_update_collect_tiles(image, image_buffer, partial_update_user);
  EXPECT_EQ(result, PARTIAL_UPDATE_CHANGES_AVAILABLE);

  result = BKE_image_partial_update_collect_tiles(image, image_buffer, partial_update_user);
  EXPECT_EQ(result, PARTIAL_UPDATE_NO_CHANGES);
  BLI_rcti_init(&region, 1023, 1024, 0, 1);
  BKE_image_partial_update_register_mark_region(image, image_buffer, &region);
  result = BKE_image_partial_update_collect_tiles(image, image_buffer, partial_update_user);
  EXPECT_EQ(result, PARTIAL_UPDATE_CHANGES_AVAILABLE);
}

TEST_F(ImagePartialUpdateTest, sequential_mark_region)
{
  ePartialUpdateCollectResult result;
  /* First tile should always return a full update. */
  result = BKE_image_partial_update_collect_tiles(image, image_buffer, partial_update_user);
  EXPECT_EQ(result, PARTIAL_UPDATE_NEED_FULL_UPDATE);
  /* Second invoke should now detect no changes. */
  result = BKE_image_partial_update_collect_tiles(image, image_buffer, partial_update_user);
  EXPECT_EQ(result, PARTIAL_UPDATE_NO_CHANGES);

  {
    /* Mark region. */
    rcti region;
    BLI_rcti_init(&region, 10, 20, 40, 50);
    BKE_image_partial_update_register_mark_region(image, image_buffer, &region);

    /* Partial Update should be available. */
    result = BKE_image_partial_update_collect_tiles(image, image_buffer, partial_update_user);
    EXPECT_EQ(result, PARTIAL_UPDATE_CHANGES_AVAILABLE);

    /* Check tiles. */
    PartialUpdateTile changed_tile;
    ePartialUpdateIterResult iter_result;
    iter_result = BKE_image_partial_update_next_tile(partial_update_user, &changed_tile);
    EXPECT_EQ(iter_result, PARTIAL_UPDATE_ITER_TILE_LOADED);
    EXPECT_EQ(BLI_rcti_inside_rcti(&changed_tile.region, &region), true);
    iter_result = BKE_image_partial_update_next_tile(partial_update_user, &changed_tile);
    EXPECT_EQ(iter_result, PARTIAL_UPDATE_ITER_NO_TILES_LEFT);

    result = BKE_image_partial_update_collect_tiles(image, image_buffer, partial_update_user);
    EXPECT_EQ(result, PARTIAL_UPDATE_NO_CHANGES);
  }

  {
    /* Mark different region. */
    rcti region;
    BLI_rcti_init(&region, 710, 720, 740, 750);
    BKE_image_partial_update_register_mark_region(image, image_buffer, &region);

    /* Partial Update should be available. */
    result = BKE_image_partial_update_collect_tiles(image, image_buffer, partial_update_user);
    EXPECT_EQ(result, PARTIAL_UPDATE_CHANGES_AVAILABLE);

    /* Check tiles. */
    PartialUpdateTile changed_tile;
    ePartialUpdateIterResult iter_result;
    iter_result = BKE_image_partial_update_next_tile(partial_update_user, &changed_tile);
    EXPECT_EQ(iter_result, PARTIAL_UPDATE_ITER_TILE_LOADED);
    EXPECT_EQ(BLI_rcti_inside_rcti(&changed_tile.region, &region), true);
    iter_result = BKE_image_partial_update_next_tile(partial_update_user, &changed_tile);
    EXPECT_EQ(iter_result, PARTIAL_UPDATE_ITER_NO_TILES_LEFT);

    result = BKE_image_partial_update_collect_tiles(image, image_buffer, partial_update_user);
    EXPECT_EQ(result, PARTIAL_UPDATE_NO_CHANGES);
  }
}

}  // namespace blender::bke::image
