
#include "gpu_shader_create_info.hh"

GPU_SHADER_CREATE_INFO(draw_view)
    .uniform_buf(0, "ViewInfos", "drw_view", Frequency::PASS)
    .uniform_buf(0, "ObjectMatrices", "drw_matrices[DRW_RESOURCE_CHUNK_LEN]", Frequency::BATCH);
