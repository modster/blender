#include "gpu_interface_info.hh"
#include "gpu_shader_create_info.hh"

GPU_SHADER_CREATE_INFO(gpu_shader_2D_uv_facedots)
    .vertex_in(0, Type::VEC2, "pos")
    .vertex_in(1, Type::INT, "flag")
    .vertex_out(smooth_color_iface)
    .fragment_out(0, Type ::VEC4, "fragColor")
    .push_constant(0, Type::MAT4, "ModelViewProjectionMatrix")
    .push_constant(16, Type::VEC4, "vertColor")
    .push_constant(20, Type::VEC4, "selectColor")
    .vertex_source("gpu_shader_2D_edituvs_facedots_vert.glsl")
    .fragment_source("gpu_shader_point_varying_color_frag.glsl")
    .do_static_compilation(true);