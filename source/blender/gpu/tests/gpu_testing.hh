#include "testing/testing.h"

#include "GHOST_C-api.h"

struct GPUContext;

namespace blender::gpu {

/* Test class that setups a GPUContext for test cases.
 *
 * Usage:
 *   TEST_F(GPUTest, my_gpu_test) {
 *     ...
 *   }
 */
class GPUTest : public ::testing::Test {
 private:
  GHOST_SystemHandle ghost_system;
  GHOST_ContextHandle ghost_context;
  struct GPUContext *context;

 protected:
  GHOST_TDrawingContextType draw_context_type = GHOST_kDrawingContextTypeOpenGL;

  void SetUp() override;
  void TearDown() override;
};

class GPUOpenGLTest : public GPUTest {
 public:
  GPUOpenGLTest();
};

#ifdef WITH_VULKAN

class GPUVulkanTest : public GPUTest {
 public:
  GPUVulkanTest();
};

#endif

}  // namespace blender::gpu
