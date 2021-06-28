/* Apache License, Version 2.0 */

#include "draw_testing.hh"

#include "GPU_shader.h"

#include "IMB_imbuf.h"

#include "BKE_appdir.h"
#include "BKE_idtype.h"

#include "DRW_engine.h"
#include "draw_manager_testing.h"

namespace blender::draw {

/* Base class for draw test cases. It will setup and tear down the GPU part around each test. */
void DrawTest::SetUp()
{
  GPUTest::SetUp();

  /* Initialize color management. Required to construct a scene creation depends on it. */
  BKE_idtype_init();
  BKE_appdir_init();
  IMB_init();

  DRW_engines_register();

  DRW_draw_state_init_gtests(GPU_SHADER_CFG_DEFAULT);
}

void DrawTest::TearDown()
{
  DRW_engines_free();

  IMB_exit();
  BKE_appdir_exit();

  GPUTest::TearDown();
}

}  // namespace blender::draw
