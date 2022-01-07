#include "gpu_interface_info.hh"
#include "gpu_shader_create_info.hh"

GPU_SHADER_CREATE_INFO(gpu_shader_2D_uv_faces_stretch)
    .vertex_in(0, Type::VEC2, "pos")
    .vertex_out(no_perspective_color_iface)
    .fragment_out(0, Type::VEC4, "fragColor")
    .push_constant(0, Type::MAT4, "ModelViewProjectionMatrix")
    .push_constant(16, Type::VEC2, "aspect")
    .vertex_source("gpu_shader_2D_edituvs_stretch_vert.glsl")
    .fragment_source("gpu_shader_2D_smooth_color_frag.glsl")
    .additional_info("gpu_srgb_to_framebuffer_space");

GPU_SHADER_CREATE_INFO(gpu_shader_2D_uv_faces_stretch_area)
    .additional_info("gpu_shader_2D_uv_faces_stretch")
    .vertex_in(1, Type::FLOAT, "ratio")
    .push_constant(20, Type::FLOAT, "totalAreaRatio")
    .push_constant(24, Type::FLOAT, "totalAreaRatioInv")
    .do_static_compilation(true);

GPU_SHADER_CREATE_INFO(gpu_shader_2D_uv_faces_stretch_angle)
    .additional_info("gpu_shader_2D_uv_faces_stretch")
    .vertex_in(1, Type::VEC2, "uv_angles")
    .vertex_in(2, Type::FLOAT, "angle")
    .define("STRETCH_ANGLE")
    .do_static_compilation(true);
