#include "gpu_shader_create_info.hh"

GPU_SHADER_CREATE_INFO(gpu_shader_3D_point_fixed_size_varying_color)
    .vertex_in(0, Type::VEC3, "pos")
    .vertex_in(1, Type::VEC4, "color")
    .vertex_out(smooth_color_iface)
    .fragment_out(0, Type::VEC4, "fragColor")
    .push_constant(0, Type::MAT4, "ModelViewProjectionMatrix")
    .push_constant(16, Type::FLOAT, "size")
    .vertex_source("gpu_shader_3D_point_fixed_size_varying_color_vert.glsl")
    .fragment_source("gpu_shader_point_varying_color_frag.glsl")
    .do_static_compilation(true);
/*
    [GPU_SHADER_3D_POINT_FIXED_SIZE_VARYING_COLOR] =
        {
            .name = "GPU_SHADER_3D_POINT_FIXED_SIZE_VARYING_COLOR",
            .vert = datatoc_gpu_shader_3D_point_fixed_size_varying_color_vert_glsl,
            .frag = datatoc_gpu_shader_point_varying_color_frag_glsl,
},
*/
GPU_SHADER_CREATE_INFO(gpu_shader_3D_point_varying_size_varying_color)
    .vertex_in(0, Type::VEC3, "pos")
    .vertex_in(1, Type::VEC4, "color")
    .vertex_in(2, Type::FLOAT, "size")
    .vertex_out(smooth_color_iface)
    .fragment_out(0, Type::VEC4, "fragColor")
    .push_constant(0, Type::MAT4, "ModelViewProjectionMatrix")
    .vertex_source("gpu_shader_3D_point_varying_size_varying_color_vert.glsl")
    .fragment_source("gpu_shader_point_varying_color_frag.glsl")
    .do_static_compilation(true);
/*
    [GPU_SHADER_3D_POINT_VARYING_SIZE_VARYING_COLOR] =
        {
            .name = "GPU_SHADER_3D_POINT_VARYING_SIZE_VARYING_COLOR",
            .vert = datatoc_gpu_shader_3D_point_varying_size_varying_color_vert_glsl,
            .frag = datatoc_gpu_shader_point_varying_color_frag_glsl,
},
*/
GPU_SHADER_CREATE_INFO(gpu_shader_3D_point_uniform_size_uniform_color_aa)
    .vertex_in(0, Type::VEC3, "pos")
    .vertex_out(smooth_radii_iface)
    .fragment_out(0, Type::VEC4, "fragColor")
    .push_constant(0, Type::MAT4, "ModelViewProjectionMatrix")
    .push_constant(16, Type::VEC4, "color")
    .push_constant(20, Type::FLOAT, "size")
    .push_constant(24, Type::FLOAT, "outlineWidth")
    .vertex_source("gpu_shader_3D_point_uniform_size_aa_vert.glsl")
    .fragment_source("gpu_shader_point_uniform_color_aa_frag.glsl")
    .do_static_compilation(true);
/*
    [GPU_SHADER_3D_POINT_UNIFORM_SIZE_UNIFORM_COLOR_AA] =
        {
            .name = "GPU_SHADER_3D_POINT_UNIFORM_SIZE_UNIFORM_COLOR_AA",
            .vert = datatoc_gpu_shader_3D_point_uniform_size_aa_vert_glsl,
            .frag = datatoc_gpu_shader_point_uniform_color_aa_frag_glsl,
},
*/