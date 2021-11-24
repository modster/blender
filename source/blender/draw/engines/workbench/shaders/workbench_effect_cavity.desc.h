
#include "gpu_shader_descriptor.h"

GPU_SHADER_DESCRIPTOR(workbench_effect_cavity)
{
    .do_static_compilation = true,
    .fragment_outputs =
        {
            [0] = FRAGMENT_OUTPUT(VEC4, "fragColor"),
        },
    .resources =
        {
            [DESCRIPTOR_SET_1] =
                {
                    [0] = SAMPLER(FLOAT_2D, "depthBuffer", GPU_SAMPLER_DEFAULT),
                    [1] = SAMPLER(FLOAT_2D, "normalBuffer", GPU_SAMPLER_DEFAULT),
                    [2] = SAMPLER(UINT_2D, "objectIdBuffer", GPU_SAMPLER_DEFAULT),
                },
        },
    .fragment_source = "workbench_effect_cavity_frag.glsl",
    .additional_descriptors =
        {
            "draw_fullscreen",
        },
};
