
#include "gpu_shader_descriptor.h"

GPU_STAGE_INTERFACE_CREATE(fullscreen_iface,
                           {
                               {VEC4, "uvcoordsvar"},
                           })

GPU_SHADER_DESCRIPTOR(draw_fullscreen,
                      {
                          .vertex_out_interfaces =
                              {
                                  [0] = STAGE_INTERFACE("", fullscreen_iface),
                              },
                          .vertex_source = "common_fullscreen_vert.glsl",
                      })
