#include "gpu_shader_create_info.hh"

GPU_SHADER_CREATE_INFO(gpu_shader_sculpt_pixel_extraction)
    .local_group_size(1, 1)
    .storage_buf(0, Qualifier::READ, "TexturePaintPolygon", "polygons[]")
    .image(1, GPU_RGBA32I, Qualifier::WRITE, ImageType::INT_2D, "pixels")
    .push_constant(Type::INT, "num_polygons")
    .compute_source("gpu_shader_sculpt_pixel_extraction_comp.glsl")
    .typedef_source("GPU_shader_shared.h")
    .do_static_compilation(true);
