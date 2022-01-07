#include "gpu_interface_info.hh"
#include "gpu_shader_create_info.hh"

GPU_SHADER_CREATE_INFO(gpu_shader_2D_uv_faces)
    .vertex_in(0, Type::VEC2, "pos")
    .vertex_in(1, Type::INT, "flag")
    .vertex_out(flat_color_iface)
    .fragment_out(0, Type::VEC4, "fragColor")
    .push_constant(0, Type::MAT4, "ModelViewProjectionMatrix")
    .push_constant(16, Type::VEC4, "faceColor")
    .push_constant(20, Type::VEC4, "selectColor")
    .push_constant(24, Type::VEC4, "activeColor")

    .vertex_source("gpu_shader_2D_edituvs_faces_vert.glsl")
    .fragment_source("gpu_shader_flat_color_frag.glsl")
    .additional_info("gpu_srgb_to_framebuffer_space")
    .do_static_compilation(true);
/*
    [GPU_SHADER_2D_UV_FACES] =
        {
            .name = "GPU_SHADER_2D_UV_FACES",
            .vert = datatoc_gpu_shader_2D_edituvs_faces_vert_glsl,
            .frag = datatoc_gpu_shader_flat_color_frag_glsl,
},
*/