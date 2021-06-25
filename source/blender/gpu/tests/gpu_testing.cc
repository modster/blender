/* Apache License, Version 2.0 */

#include "testing/testing.h"

#include "BKE_global.h"

#include "GPU_context.h"
#include "GPU_init_exit.h"
#include "gpu_testing.hh"

#include "GHOST_C-api.h"

namespace blender::gpu {

void GPUTest::SetUp()
{
  GHOST_GLSettings glSettings = {0};
  ghost_system = GHOST_CreateSystem();
  ghost_context = GHOST_CreateOpenGLContext(ghost_system, draw_context_type, glSettings);
  context = GPU_context_create(NULL, ghost_context);
  GPU_init();
}

void GPUTest::TearDown()
{
  GPU_exit();
  GPU_backend_exit();
  GPU_context_discard(context);
  GHOST_DisposeOpenGLContext(ghost_system, ghost_context);
  GHOST_DisposeSystem(ghost_system);
}

GPUOpenGLTest::GPUOpenGLTest()
{
  G.debug &= ~G_DEBUG_VK_CONTEXT;
  draw_context_type = GHOST_kDrawingContextTypeOpenGL;
}

#ifdef WITH_VULKAN

GPUVulkanTest::GPUVulkanTest()
{
  G.debug |= G_DEBUG_VK_CONTEXT;
  draw_context_type = GHOST_kDrawingContextTypeVulkan;
}

#endif

}  // namespace blender::gpu
