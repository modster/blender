#include "gpu_interface_info.hh"
#include "gpu_shader_create_info.hh"

GPU_SHADER_INTERFACE_INFO(smooth_radii_fill_color_outline_iface, "")
    .smooth(Type::VEC4, "fillColor")
    .smooth(Type::VEC4, "outlineColor")
    .smooth(Type::VEC4, "radii");

GPU_SHADER_CREATE_INFO(gpu_shader_2D_uv_verts)
    .vertex_in(0, Type::VEC2, "pos")
    .vertex_in(1, Type::INT, "flag")
    .vertex_out(smooth_radii_fill_color_outline_iface)
    .fragment_out(0, Type::VEC4, "fragColor")
    .push_constant(0, Type::MAT4, "ModelViewProjectionMatrix")
    .push_constant(16, Type::VEC4, "vertColor")
    .push_constant(20, Type::VEC4, "selectColor")
    .push_constant(24, Type::VEC4, "pinnedColor")
    .push_constant(28, Type::FLOAT, "pointSize")
    .push_constant(29, Type::FLOAT, "outlineWidth")
    .vertex_source("gpu_shader_2D_edituvs_points_vert.glsl")
    .fragment_source("gpu_shader_point_varying_color_varying_outline_aa_frag.glsl")
    .additional_info("gpu_srgb_to_framebuffer_space")
    .do_static_compilation(true);

GPU_SHADER_CREATE_INFO(gpu_shader_2D_uv_uniform_color)
    .vertex_in(0, Type::VEC2, "u")
    .fragment_out(0, Type::VEC4, "fragColor")
    .vertex_source("gpu_shader_2D_vert.glsl")
    .fragment_source("gpu_shader_uniform_color_frag.glsl")
    .push_constant(0, Type::MAT4, "ModelViewProjectionMatrix")
    .push_constant(16, Type::VEC4, "color")
    .define("pos u")
    .additional_info("gpu_srgb_to_framebuffer_space")
    .do_static_compilation(true);