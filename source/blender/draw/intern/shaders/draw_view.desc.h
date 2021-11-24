
#include "gpu_shader_descriptor.h"

GPU_SHADER_DESCRIPTOR(draw_view)
{
    .resources =
        {
            [DESCRIPTOR_SET_1] =
                {
                    [0] = UNIFORM_BUFFER("ViewInfos", "drw_view"),
                    [1] = UNIFORM_BUFFER("ObjectMatrices", "drw_matrices[DRW_RESOURCE_CHUNK_LEN]"),
                },
        },
};
