
#include "gpu_shader_descriptor.h"

GPU_SHADER_DESCRIPTOR(gpu_clip_planes,
                      {
                          .resources =
                              {
                                  [DESCRIPTOR_SET_0] =
                                      {
                                          [1] = UNIFORM_BUFFER("GPUClipPlanes", "clipPlanes"),
                                      },
                              },
                          .defines =
                              {
                                  [5] = "#define USE_WORLD_CLIP_PLANES\n",
                              },
                      })
