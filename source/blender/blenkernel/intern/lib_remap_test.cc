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

#include "BLI_utildefines.h"

#include "CLG_log.h"

#include "DNA_node_types.h"
#include "DNA_scene_types.h"

#include "RNA_define.h"

#include "BKE_appdir.h"
#include "BKE_context.h"
#include "BKE_global.h"
#include "BKE_idtype.h"
#include "BKE_lib_id.h"
#include "BKE_lib_remap.h"
#include "BKE_main.h"
#include "BKE_node.h"
#include "BKE_scene.h"

#include "IMB_imbuf.h"

#include "ED_node.h"

#include "MEM_guardedalloc.h"

namespace blender::bke::tests {

struct Context {
  Main *bmain = nullptr;
  Scene *scene = nullptr;
  bNodeTree *composite_nodetree = nullptr;
  struct bContext *C = nullptr;

  Context()
  {
    CLG_init();
    BKE_idtype_init();
    RNA_init();
    BKE_node_system_init();
    BKE_appdir_init();
    IMB_init();

    bmain = BKE_main_new();
    /* TODO(jbakker): node_composit_poll_rlayers uses G.main directly. Should be refactored. */
    G.main = bmain;
    C = CTX_create();
    CTX_data_main_set(C, bmain);
    init_test_data();
  }

  ~Context()
  {
    BKE_main_free(bmain);
    CTX_free(C);
    G.main = nullptr;
    bmain = nullptr;
    C = nullptr;
    scene = nullptr;
    BKE_node_system_exit();
    RNA_exit();
    IMB_exit();
    BKE_appdir_exit();
    CLG_exit();
  }

  void init_test_data()
  {
    add_scene();
    add_composite();
  }

  void add_scene()
  {
    scene = BKE_scene_add(bmain, "IDRemapScene");
    CTX_data_scene_set(C, scene);
  }

  void add_composite()
  {
    ED_node_composit_default(C, scene);
    composite_nodetree = scene->nodetree;
  }
};

TEST(lib_remap, embedded_ids_can_not_be_remapped)
{
  Context context;
  bNodeTree *other_tree = static_cast<bNodeTree *>(BKE_id_new_nomain(ID_NT, nullptr));

  EXPECT_NE(context.scene, nullptr);
  EXPECT_NE(context.composite_nodetree, nullptr);
  EXPECT_EQ(context.composite_nodetree, context.scene->nodetree);

  BKE_libblock_remap(context.bmain, context.composite_nodetree, other_tree, 0);

  EXPECT_EQ(context.composite_nodetree, context.scene->nodetree);
  EXPECT_NE(context.scene->nodetree, other_tree);

  BKE_id_free(nullptr, other_tree);
}

TEST(lib_remap, embedded_ids_can_not_be_deleted)
{
  Context context;

  EXPECT_NE(context.scene, nullptr);
  EXPECT_NE(context.composite_nodetree, nullptr);
  EXPECT_EQ(context.composite_nodetree, context.scene->nodetree);

  BKE_libblock_remap(context.bmain, context.composite_nodetree, nullptr, 0);

  EXPECT_EQ(context.composite_nodetree, context.scene->nodetree);
  EXPECT_NE(context.scene->nodetree, nullptr);
}

}  // namespace blender::bke::tests
