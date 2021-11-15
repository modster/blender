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
static Image *create_test_image(Main *bmain, int width, int height)
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

static void init()
{
  CLG_init();
  BKE_idtype_init();
  BKE_appdir_init();
  IMB_init();
}

static void deinit()
{
  IMB_exit();
  BKE_appdir_exit();
  CLG_exit();
}

TEST(image_partial_update, mark_full_update)
{
  init();
  Main *bmain = BKE_main_new();
  Image *image = create_test_image(bmain, 1024, 1024);
  ImBuf *image_buffer = BKE_image_acquire_ibuf(image, nullptr, nullptr);

  struct PartialUpdateUser *partial_update_user = BKE_image_partial_update_create(image);
  EXPECT_NE(partial_update_user, nullptr);

  ePartialUpdateCollectResult result;
  /* First tile should always return a full update. */
  result = BKE_image_partial_update_collect_tiles(image, partial_update_user);
  EXPECT_EQ(result, PARTIAL_UPDATE_NEED_FULL_UPDATE);
  /* Second invoke should now detect no changes. */
  result = BKE_image_partial_update_collect_tiles(image, partial_update_user);
  EXPECT_EQ(result, PARTIAL_UPDATE_NO_CHANGES);

  /* Mark full update */
  BKE_image_partial_update_register_mark_full_update(image, image_buffer);

  /* Validate need full update followed by no changes. */
  result = BKE_image_partial_update_collect_tiles(image, partial_update_user);
  EXPECT_EQ(result, PARTIAL_UPDATE_NEED_FULL_UPDATE);
  result = BKE_image_partial_update_collect_tiles(image, partial_update_user);
  EXPECT_EQ(result, PARTIAL_UPDATE_NO_CHANGES);

  BKE_image_release_ibuf(image, image_buffer, nullptr);

  BKE_image_partial_update_free(partial_update_user);

  BKE_main_free(bmain);
  deinit();
}

}  // namespace blender::bke::image

