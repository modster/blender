
#include "gpu_shader_create_info.hh"

GPU_SHADER_CREATE_INFO(gpu_clip_planes)
    .uniform_buf(1, "GPUClipPlanes", "clipPlanes", Frequency::PASS)
    .typedef_source("GPU_shader_shared.h")
    .define("USE_WORLD_CLIP_PLANES");
