
#include "gpu_shader_create_info.hh"

GPU_SHADER_CREATE_INFO(draw_view)
    .uniform_buf(0, "ViewInfos", "drw_view", Frequency::PASS)
    .uniform_buf(8, "ObjectMatrices", "drw_matrices[DRW_RESOURCE_CHUNK_LEN]", Frequency::BATCH);

GPU_SHADER_CREATE_INFO(draw_view_instanced_attr)
    .uniform_buf(0, "ViewInfos", "drw_view", Frequency::PASS)
    .push_constant(0, Type::MAT4, "ModelMatrix")
    .push_constant(16, Type::MAT4, "ModelMatrixInverse");
