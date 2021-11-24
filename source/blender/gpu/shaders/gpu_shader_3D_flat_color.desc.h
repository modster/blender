
#include "gpu_shader_descriptor.h"

GPU_STAGE_INTERFACE_CREATE(flat_color_iface,
                           {
                               {VEC4, "finalColor", FLAT},
                           })

GPU_SHADER_DESCRIPTOR(gpu_shader_3D_flat_color,
                      {
                          .vertex_inputs =
                              {
                                  [0] = VERTEX_INPUT(VEC3, "pos"),
                                  [1] = VERTEX_INPUT(VEC4, "col"),
                              },
                          .vertex_out_interfaces =
                              {
                                  [0] = STAGE_INTERFACE("", flat_color_iface),
                              },
                          .fragment_outputs =
                              {
                                  [0] = FRAGMENT_OUTPUT(VEC4, "fragColor"),
                              },
                          .resources =
                              {
                                  [DESCRIPTOR_SET_0] =
                                      {
                                          // [0] = common->matrix_stack_block,
                                      },
                              },
                          .push_constants =
                              {
                                  [1] = PUSH_CONSTANT(BOOL, "srgbTarget"),
                              },
                          .vertex_source = "gpu_shader_3D_flat_color_vert.glsl",
                          .fragment_source = "gpu_shader_flat_color_frag.glsl",
                      })

GPU_SHADER_DESCRIPTOR(gpu_shader_3D_flat_color_clipped,
                      {
                          .additional_descriptors =
                              {
                                  &gpu_shader_3D_flat_color,
                                  &gpu_clip_planes,
                              },
                      })
