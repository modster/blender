
#include "gpu_shader_descriptor.h"

GPU_SHADER_DESCRIPTOR(draw_object_infos)
{
    .resources =
        {
            [DESCRIPTOR_SET_1] =
                {
                    [2] = UNIFORM_BUFFER("ObjectInfos", "drw_infos[DRW_RESOURCE_CHUNK_LEN]"),
                },
        },
};
