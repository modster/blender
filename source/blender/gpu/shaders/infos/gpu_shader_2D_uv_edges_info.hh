#include "gpu_interface_info.hh"
#include "gpu_shader_create_info.hh"

GPU_SHADER_INTERFACE_INFO(flat_edituvs_edges_iface, "")
    .flat(Type::VEC4, "finalColor")
    .no_perspective(Type::VEC2, "stipple_pos")
    .flat(Type::VEC2, "stipple_start");
GPU_SHADER_INTERFACE_INFO(no_perspective_edituvs_edges_iface, "")
    .no_perspective(Type::VEC4, "finalColor")
    .no_perspective(Type::VEC2, "stipple_pos")
    .flat(Type::VEC2, "stipple_start");

GPU_SHADER_CREATE_INFO(gpu_shader_2D_uv_edges)
    .vertex_in(0, Type::VEC2, "pos")
    .vertex_in(1, Type::INT, "flag")
    .vertex_out(flat_edituvs_edges_iface)
    .fragment_out(0, Type::VEC4, "fragColor")
    .push_constant(0, Type::MAT4, "ModelViewProjectionMatrix")
    .push_constant(16, Type::VEC4, "edgeColor")
    .push_constant(20, Type::VEC4, "selectColor")
    .push_constant(24, Type::FLOAT, "dashWidth")

    .vertex_source("gpu_shader_2D_edituvs_edges_vert.glsl")
    .fragment_source("gpu_shader_2D_edituvs_edges_frag.glsl")
    .additional_info("gpu_srgb_to_framebuffer_space")
    .do_static_compilation(true);

GPU_SHADER_CREATE_INFO(gpu_shader_2D_uv_edges_smooth)
    .vertex_in(0, Type::VEC2, "pos")
    .vertex_in(1, Type::INT, "flag")
    .vertex_out(no_perspective_edituvs_edges_iface)
    .fragment_out(0, Type::VEC4, "fragColor")
    .push_constant(0, Type::MAT4, "ModelViewProjectionMatrix")
    .push_constant(16, Type::VEC4, "edgeColor")
    .push_constant(20, Type::VEC4, "selectColor")
    .push_constant(24, Type::FLOAT, "dashWidth")

    .vertex_source("gpu_shader_2D_edituvs_edges_vert.glsl")
    .fragment_source("gpu_shader_2D_edituvs_edges_frag.glsl")
    .define("SMOOTH_COLOR")
    .additional_info("gpu_srgb_to_framebuffer_space")
    .do_static_compilation(true);
