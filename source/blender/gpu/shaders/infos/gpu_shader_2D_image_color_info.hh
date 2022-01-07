#include "gpu_shader_create_info.hh"

GPU_SHADER_CREATE_INFO(gpu_shader_2D_image_color)
    .additional_info("gpu_shader_2D_image_common")
    .push_constant(16, Type::VEC4, "color")
    .fragment_source("gpu_shader_image_color_frag.glsl")
    .do_static_compilation(true);
